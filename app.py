import gradio as gr

from simplet5 import SimpleT5
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")

def summarize(text_input):

  if text_input == None:
    return None

  model.load_model("simplet5-epoch-5-train-loss-0.6859", use_gpu=False)
  text_to_summarize = """summarize: """ + text_input
  predicted_output = model.predict(text_to_summarize)
  return predicted_output


text_i = gr.inputs.Textbox(lines=10, placeholder="Enter text", label="Text")

iface = gr.Interface(fn = summarize, inputs = text_i, outputs="textbox")

iface.launch()
