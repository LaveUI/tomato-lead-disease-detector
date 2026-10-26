import argparse
import gradio as gr
from fieldready.inference import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='artifacts/fieldready')
    parser.add_argument('--share', action='store_true', help='Create a public Gradio link (e.g. Colab)')
    args = parser.parse_args()
    predictor = Predictor(args.model_dir)

    def screen(image):
        result = predictor.predict(image)
        return result['message'], result['scores']

    demo = gr.Interface(
        fn=screen, inputs=gr.Image(type='pil', label='Upload a tomato leaf photo'),
        outputs=[gr.Textbox(label='Screening result'), gr.Label(num_top_classes=len(predictor.classes), label='Model scores (not diagnostic certainty)')],
        title='FieldReady — Reliable Tomato Leaf Disease Detection',
        description='A screening aid, not a definitive diagnosis. Photograph one tomato leaf clearly in natural light. '
                    'Screens for Healthy, Early Blight, and Late Blight. A learned other-disease category, including Leaf Mold, '
                    'returns Uncertain. Uncertain results require a clearer photo or expert review.',
        article='Softmax scores do not guarantee correctness. Other diseases, non-tomato plants, and poor photos '
                'can receive confident incorrect predictions. Consult a qualified expert before treatment.',
        flagging_mode='never')
    demo.launch(share=args.share)


if __name__ == '__main__':
    main()
