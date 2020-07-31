from numpy.random import uniform
from numpy import where
from plotnine import ggplot, aes, geom_point, scale_color_brewer, facet_wrap, coord_fixed
from pandas import DataFrame, Categorical, concat
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
import plotly.express as px

# Create toy dataset
N = 5000
x = uniform(-1, 1, N)
y = uniform(-1, 1, N)
z = where(x * y > 0, 1, 0)
xor = DataFrame({'x':x, 'y':y, 'class':z})
xor['class'] = xor['class'].astype("category")

# Plot data
g = ggplot(xor, aes(x="x", y="y", color="class")) +\
    geom_point(size=1) +\
    scale_color_brewer(type="qual", palette="Set1") +\
    coord_fixed()
g.draw()
g.save("xor.png", height=6, width=6)

# Split data for training and test
X, y = xor[["x","y"]].to_numpy(), xor["class"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.25)

# Build model
def build_binary_mlp(input_dim, hidden):
    """Create MLP for binary classification

    Args:
        input_dim (int): number of input variables.
        hidden (Dict): hidden layer name as key and number of units as value. 
            Allow multiple layers.
    """
    
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for (layer_name, units) in hidden.items():
        model.add(Dense(units, activation="tanh", name=layer_name))
    model.add(Dense(1, activation="sigmoid", name="out"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

model = build_binary_mlp(X_tr.shape[1], {'h1':8, 'h2':2})
model.summary()
plot_model(model, show_shapes=True, to_file='nn_model.png')

# Train model
def extract_layer(model, layer, X):
    """Extract layer output

    Args:
        model (Sequential): keras fitted DL model.
        layer (str): Layer name.
        X (ndarray): input data to processed.
    """

    layer_extractor = Model(
        inputs  = model.input, 
        outputs = model.get_layer(layer).output
    )
    out = DataFrame(layer_extractor(X).numpy(), columns=['h1','h2'])
    
    return(out)

epochs = 30
data = []
for i in range(0, epochs+1): 
    out = extract_layer(model, "h2", X_te)
    out["class"] = Categorical(y_te)
    out["epoch"] = i
    data.append(out)
    model.fit(
        x               = X_tr, 
        y               = y_tr,
        batch_size      = 32,
        epochs          = 1,
        validation_data = (X_te, y_te)
    )

# Create animation
anim = concat(data)
fig = px.scatter(anim, x="h1", y="h2", color="class", 
                 animation_frame="epoch", width=600, height=600,
                 range_x=[-1.1, 1.1], range_y=[-1.1, 1.1])
fig.show()
fig.write_html("docs/index.html")

# Plot layer output
g = ggplot(anim, aes(x="h1", y="h2", color="class")) +\
    geom_point(size=0.1) +\
    scale_color_brewer(type="qual", palette="Set1") +\
    coord_fixed() +\
    facet_wrap('epoch', ncol=6)
g.draw()
g.save("hidden_output.png", height=10, width=10)