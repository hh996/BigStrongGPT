import streamlit as st
from threading import Thread
import torch
from transformers import TextIteratorStreamer, AutoTokenizer

# 自定义模型导入
from model.model_big_strong import BigStrongForCausalLM, BigStrongConfig

# RAG 相关库
import os
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
import hashlib
from datetime import datetime

# 全局变量
MODEL_PATH = "../model/full_sft_512.pth"
TOKENIZER_PATH = "../model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOCUMENT_FOLDER = "../doc"  # 文档路径
VECTOR_DB_FOLDER = "../vector_db"  # 向量数据库存储路径


@st.cache_resource
def load_model():
    status_text = st.empty()
    status_text.text("正在加载语言模型...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    # 对齐分词器与模型的重要标识符，避免生成立即结束
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    config = BigStrongConfig(
        hidden_size=512,
        num_hidden_layers=8,
        bos_token_id=bos_id if bos_id is not None else 1,
        eos_token_id=eos_id if eos_id is not None else 2,
        vocab_size=vocab_size,
    )
    model = BigStrongForCausalLM(config)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()
    # 设置生成配置，确保与 tokenizer 对齐
    try:
        model.generation_config.pad_token_id = (
            pad_id if pad_id is not None else config.eos_token_id
        )
        model.generation_config.bos_token_id = config.bos_token_id
        model.generation_config.eos_token_id = config.eos_token_id
    except Exception:
        pass

    status_text.text("语言模型加载完成！")
    status_text.empty()
    return model, tokenizer


def get_documents_hash(document_folder):
    """获取文件名和修改时间的哈希值来判断文档是否有变更"""
    hash_md5 = hashlib.md5()

    if not os.path.exists(document_folder):
        return ""

    # 获取所有支持的文件
    supported_exts = {".txt", ".pdf", ".docx"}
    files_info = []

    for file_name in sorted(os.listdir(document_folder)):
        file_path = os.path.join(document_folder, file_name)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in supported_exts:
                # 记录文件名和修改时间
                mtime = os.path.getmtime(file_path)
                files_info.append(f"{file_name}:{mtime}")

    # 计算整体哈希
    files_str = "|".join(files_info)
    hash_md5.update(files_str.encode("utf-8"))
    return hash_md5.hexdigest()


def save_vector_db(documents, embedding_model_name, index, vector_db_folder, docs_hash):
    """保存向量数据库到磁盘"""
    os.makedirs(vector_db_folder, exist_ok=True)

    # 保存文档列表
    with open(os.path.join(vector_db_folder, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)

    # 保存 FAISS 索引
    faiss.write_index(index, os.path.join(vector_db_folder, "faiss_index.bin"))

    # 保存元数据
    metadata = {
        "embedding_model": embedding_model_name,
        "documents_count": len(documents),
        "docs_hash": docs_hash,
        "created_time": datetime.now().isoformat(),
        "dimension": index.d,  # 向量维度
    }

    with open(
        os.path.join(vector_db_folder, "metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    st.success(f"向量数据库已保存到：{vector_db_folder}")


# --- 加载向量数据库 ---
def load_vector_db(vector_db_folder, embedding_model_name):
    """从磁盘加载向量数据库"""
    try:
        # 检查必要文件是否存在
        required_files = ["documents.pkl", "faiss_index.bin", "metadata.json"]
        for file_name in required_files:
            if not os.path.exists(os.path.join(vector_db_folder, file_name)):
                return None, None, None

        # 加载元数据
        with open(
            os.path.join(vector_db_folder, "metadata.json"), "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        # 检查模型是否匹配
        if metadata.get("embedding_model") != embedding_model_name:
            st.warning(
                f"向量数据库使用的模型 ({metadata.get('embedding_model')}) 与当前模型 ({embedding_model_name}) 不匹配，将重新构建"
            )
            return None, None, None

        # 加载文档列表
        with open(os.path.join(vector_db_folder, "documents.pkl"), "rb") as f:
            documents = pickle.load(f)

        # 加载 FAISS 索引
        index = faiss.read_index(os.path.join(vector_db_folder, "faiss_index.bin"))

        st.info(
            f"成功加载向量数据库：{len(documents)} 个文档块，创建时间：{metadata.get('created_time')}"
        )
        return documents, index, metadata.get("docs_hash")

    except Exception as e:
        st.error(f"加载向量数据库失败：{e}")
        return None, None, None


@st.cache_resource
def initialize_rag(
    document_folder=DOCUMENT_FOLDER,
    vector_db_folder=VECTOR_DB_FOLDER,
    force_rebuild=False,
):
    # 中文 embedding 模型（名称实际是英文模型，但通用性强）
    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name)

    # 计算当前文档的哈希值
    current_docs_hash = get_documents_hash(document_folder)

    # 如果不强制重建，尝试加载已保存的向量库
    if not force_rebuild:
        documents, index, saved_docs_hash = load_vector_db(
            vector_db_folder, embedding_model_name
        )
        # 如果向量库存在且文档未变化，直接复用
        if documents is not None and saved_docs_hash == current_docs_hash:
            st.success("✅ 使用现有向量数据库，文档无变更")
            return documents, embedding_model, index
        elif documents is not None and saved_docs_hash != current_docs_hash:
            st.info("📝 检测到文档内容变化，重新构建向量数据库...")
        else:
            st.info("📁 未找到现有向量数据库，开始构建...")
    else:
        st.warning("🔧 强制重建向量数据库（force_rebuild=True）")

    # ========== 构建新的向量数据库 ==========
    st.write("🔨 开始构建新的向量数据库...")

    documents = []
    supported_exts = {".txt", ".pdf", ".docx"}

    # 加载supported_exts类的文档
    files_to_process = [
        f
        for f in os.listdir(document_folder)
        if os.path.isfile(os.path.join(document_folder, f))
        and os.path.splitext(f)[1].lower() in supported_exts
    ]

    if len(files_to_process) == 0:
        st.warning("⚠️ 未在文件夹中找到支持的文档（.txt/.pdf/.docx）")
        return None, None, None

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_name in enumerate(files_to_process):
        file_path = os.path.join(document_folder, file_name)
        status_text.text(f"正在处理: {file_name}")

        text = load_document(file_path)
        if text.strip():  # 只处理非空文档
            chunks = split_text(text, chunk_size=256, overlap=32)
            # 为每个块添加来源信息
            chunks_with_source = [f"[来源: {file_name}]\n{chunk}" for chunk in chunks]
            documents.extend(chunks_with_source)

        progress_bar.progress((i + 1) / len(files_to_process))

    progress_bar.empty()
    status_text.empty()

    if len(documents) == 0:
        st.warning("⚠️ 所有文档均为空或无法读取，无法构建向量数据库")
        return None, None, None

    st.write(
        f"📊 共处理 {len(files_to_process)} 个文件，生成 {len(documents)} 个文本块"
    )

    # 生成向量并构建 FAISS 索引
    with st.spinner("🧮 正在生成文本向量..."):
        embeddings = embedding_model.encode(
            documents, convert_to_numpy=True, show_progress_bar=True
        )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 保存向量库（含当前文档哈希）
    save_vector_db(
        documents, embedding_model_name, index, vector_db_folder, current_docs_hash
    )

    st.success("✅ RAG 向量数据库构建完成！")
    return documents, embedding_model, index


def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"读取文件失败：{file_path}，错误：{e}")
    return text


def split_text(text, chunk_size=256, overlap=32):
    """相邻块之间有32个词的重叠，避免关键信息被分割"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break
    return chunks


def retrieve_relevant_context(query, documents, embedding_model, index, top_k=3):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)  # 将查询转为向量
    scores, indices = index.search(query_vec, top_k)

    context_parts = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        # 添加相似度分数信息
        similarity = 1 / (1 + score)  # 转换为相似度分数
        context_parts.append(
            f"[参考资料 {i + 1}] (相似度: {similarity:.3f})\n{documents[idx]}"
        )

    return "\n\n".join(context_parts)


def get_model_response(
    prompt, max_new_tokens=512, temperature=0.85, top_p=0.8, use_rag=True
):
    history_messages = st.session_state.messages
    if use_rag:
        search_query = prompt
        if len(st.session_state.messages) >= 2:
            recent_context = []
            for msg in st.session_state.messages[-4:]:
                recent_context.append(f"{msg['role']}: {msg['content']}")
            search_query = (
                f"对话上下文：{' | '.join(recent_context)}\n当前问题：{prompt}"
            )

        context = retrieve_relevant_context(
            search_query, docs, emb_model, faiss_index, top_k=2
        )
        rag_messages = {
            "role": "user",
            "content": f"""你是一个智能助手，请参考以下资料回答问题。若资料中无相关信息，请说明"根据现有资料无法回答"。\n【参考资料】\n{context}\n请基于以上资料和对话历史，自然地回答用户的问题{prompt}。""",
        }
        final_messages = history_messages + [rag_messages]
        final_prompt = tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 非RAG模式，直接使用对话历史
        final_messages = history_messages + [{"role": "user", "content": prompt}]
        final_prompt = tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(
        final_prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(DEVICE)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    pad_id = (
        getattr(model.generation_config, "pad_token_id", None)
        or getattr(tokenizer, "pad_token_id", None)
        or getattr(model.generation_config, "eos_token_id", None)
    )
    eos_id = getattr(model.generation_config, "eos_token_id", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 8,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "no_repeat_ngram_size": 3,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        # 逐步将生成内容返回给调用方（用于 st.write_stream 实时显示）
        yield text


# --- 页面配置 ---
st.set_page_config(
    page_title="BigStrongGPT-RAG",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🧠 BigStrongGPT + RAG 智能问答系统")
st.markdown("基于本地文档进行检索增强生成，支持 `.txt`, `.pdf`, `.docx` 文件")

# --- 侧边栏：RAG 控制 ---
with st.sidebar:
    st.header("⚙️ 设置")
    use_rag = st.checkbox(
        "启用 RAG 检索增强", value=True, help="关闭则使用原始模型回答"
    )
    temperature = st.slider("Temperature", 0.1, 1.5, 0.85, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.8, 0.05)
    max_new_tokens = st.slider("最大生成长度", 128, 1024, 512, 64)

    st.divider()
    st.header("📚 向量数据库管理")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 重新构建", help="强制重新构建向量数据库"):
            st.cache_resource.clear()
            initialize_rag(force_rebuild=True)
            st.success("向量数据库已重新构建！")

    with col2:
        if st.button("🗑️ 清理缓存", help="清理所有缓存"):
            st.cache_resource.clear()
            st.success("缓存已清理！")

    # 显示向量数据库状态
    if os.path.exists(VECTOR_DB_FOLDER) and os.path.exists(
        os.path.join(VECTOR_DB_FOLDER, "metadata.json")
    ):
        try:
            with open(
                os.path.join(VECTOR_DB_FOLDER, "metadata.json"), "r", encoding="utf-8"
            ) as f:
                metadata = json.load(f)

            st.info(
                f"""
            📊 **向量数据库状态**
            - 文档块数量: {metadata.get('documents_count', 'N/A')}
            - 向量维度: {metadata.get('dimension', 'N/A')}  
            - 创建时间: {metadata.get('created_time', 'N/A')[:19]}
            """
            )
        except Exception as e:
            raise e


model, tokenizer = load_model()
docs, emb_model, faiss_index = initialize_rag(force_rebuild=False)

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 处理用户输入 ---
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手回复
    with st.chat_message("assistant"):
        with st.spinner("🧠 生成中..."):
            response = st.write_stream(
                get_model_response(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_rag=use_rag,
                )
            )

    # 保存对话
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
