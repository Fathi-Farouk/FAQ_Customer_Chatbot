import gradio as gr
from rag_pipeline_02 import build_rag_chain

# =========================
# Chain Cache (IMPORTANT)
# =========================
chain_cache = {}

def get_chain(source, db_type):
    key = f"{source}_{db_type}"

    if key not in chain_cache:
        print(f"🚀 Loading chain for {key}")
        chain_cache[key] = build_rag_chain(source, db_type)

    return chain_cache[key]


# =========================
# Chat Function
# =========================
def chat_fn(message, history, source, db_type):

    if history is None:
        history = []

    # ✅ Use cached chain (fix repeated answer issue)
    chain = get_chain(source, db_type)

    response = chain.invoke(message)
    response = str(response)

    # ✅ Gradio message format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, history


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="FAQ RAG Assistant") as demo:

    # ✅ Title (Centered + Blue)
    gr.Markdown("""
    <h2 style='text-align: center; color: #87CEEB;'>
        <span  style='font-size: 60px; font-weight: bold;'>🤖 FAQ Customer Chatbot </span>
        <span style='color: purple; font-size: 40px; font-weight: bold;'> (for STC and WE Customers) </span>
    </h2>
    """)

    # ✅ Controls Row
    with gr.Row():

        with gr.Column():
            gr.Markdown("<b style='font-size:18px;'>Select Customer</b>")
            source = gr.Dropdown(
                choices=["stc", "we"],
                value="stc",
                label=""
            )

        with gr.Column():
            gr.Markdown("<b style='font-size:18px;'>Vector DB</b>")
            db_type = gr.Dropdown(
                choices=["faiss", "chroma"],
                value="faiss",
                label=""
            )
    # ✅ Center Logo
    logo = gr.Image(
        value="assets/stc.png",
        height=300,
        width=300,
        show_label=False
    )

    def update_logo(source):
        return "assets/stc.png" if source == "stc" else "assets/we.png"

    # ✅ Chatbot (full width)
    chatbot = gr.Chatbot(height=300)

    # ✅ Input + Submit in SAME ROW
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask your question...",
            label="",
            scale=8
        )

        submit_btn = gr.Button("Submit", variant="primary", scale=1)

    # ✅ Clear button
    clear = gr.Button("Clear Chat")

    # =========================
    # 🔗 Submit logic
    # =========================
    submit_event = submit_btn.click(
        fn=chat_fn,
        inputs=[msg, chatbot, source, db_type],
        outputs=[chatbot, chatbot]
    )

    # ✅ Enable Enter key ALSO
    msg.submit(
        fn=chat_fn,
        inputs=[msg, chatbot, source, db_type],
        outputs=[chatbot, chatbot]
    )

    # ✅ Clear textbox after submit
    submit_event.then(
        lambda: "",
        None,
        msg
    )

    # ✅ Clear chat
    clear.click(lambda: [], None, chatbot)

    # ✅ Update logo
    source.change(update_logo, source, logo)
# =========================
# Run App
# =========================
if __name__ == "__main__":
    demo.launch()