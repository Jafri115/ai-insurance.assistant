"""Gradio demo app for the Insurance AI Assistant.

Launches a chat UI that uses the locally fine-tuned LoRA adapter.
This mirrors the notebook app with a streamlined layout.
"""
import argparse
import logging
from typing import List, Tuple

import gradio as gr

from inference_utils import load_model_and_tokenizer, generate_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_app")


def make_prompt(tokenizer, history: List[Tuple[str, str]], new_message: str, instruction: str) -> str:
    # Build messages list compatible with chat_template
    messages = [{"role": "system", "content": instruction}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": new_message})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback - very simple format
        convo = []
        for u, a in history:
            convo.append(f"User: {u}\nAssistant: {a}")
        convo.append(f"User: {new_message}\nAssistant:")
        return "\n\n".join(convo)


def main():
    parser = argparse.ArgumentParser(description="Insurance Assistant Demo App")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter_path", type=str, default="models/insurance-assistant-gpu")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--share", action="store_true", help="Enable public sharing link")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model, adapter_path=args.adapter_path, load_in_4bit=not args.no_4bit
    )

    system_instruction = (
        "You are a helpful insurance assistant. Answer clearly, professionally, and stay on topic."
    )

    def respond(message, history, max_new_tokens, temperature, top_p):
        history = history or []
        prompt = make_prompt(tokenizer, history, message, system_instruction)
        output = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
        )
        history.append([message, output])
        return "", history

    with gr.Blocks(title="Insurance AI Assistant") as demo:
        gr.Markdown("# ️ Insurance AI Assistant\nAsk anything about insurance.")
        chatbot = gr.Chatbot(height=520)
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask about policies, premiums, claims...", scale=4)
            send = gr.Button("Send", variant="primary")
        with gr.Accordion("Generation Settings", open=False):
            max_new_tokens = gr.Slider(32, 512, value=256, step=8, label="Max New Tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P")
        clear = gr.Button("Clear Chat")

        inputs = [msg, chatbot, max_new_tokens, temperature, top_p]
        outputs = [msg, chatbot]
        send.click(respond, inputs=inputs, outputs=outputs)
        msg.submit(respond, inputs=inputs, outputs=outputs)
        clear.click(lambda: (None, None), outputs=[msg, chatbot])

    demo.queue()
    demo.launch(share=args.share, server_port=args.server_port, server_name=args.server_name)


if __name__ == "__main__":
    main()
