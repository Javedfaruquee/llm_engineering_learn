import os
from pathlib import Path

from visualizer_plot import TokenPredictor, create_token_graph, visualize_predictions

message = "In one sentence, describe the color orange to someone who has never been able to see"
model_name = "gpt-4.1-mini"

# Next to this script, whichever directory it is run from
output_file = Path(__file__).resolve().with_name("token_predictions.png")

print(f"Asking {model_name} and recording each token's probability ...")
predictor = TokenPredictor(model_name)
predictions = predictor.predict_tokens(message)
print(f"Got {len(predictions)} tokens: {''.join(p['token'] for p in predictions)}")

G = create_token_graph(model_name, predictions)
plt = visualize_predictions(G)

# The graph is much taller than any screen, and a plt.show() window squeezes
# it to fit until the nodes and labels pile on top of each other. Save it at
# full size instead and open the image, which can be scrolled and zoomed.
plt.savefig(output_file, dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved the graph to {output_file}")

if hasattr(os, "startfile"):  # Windows: open in the default image viewer
    os.startfile(output_file)
