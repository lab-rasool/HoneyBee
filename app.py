import base64
from io import StringIO

import pandas as pd
import plotly.express as px
import torch
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dcc import send_file
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

# Initialize Dash app
app = Dash(__name__)
app.title = "Interactive t-SNE Visualization"
global_tsne_df = None
global_df_with_embeddings = None


# HuggingFace embedding model
class HuggingFaceEmbedder:
    def __init__(
        self,
        model_name,
        device=None,
        pooling_method="pooler_output",
        max_length=512,
        use_fast_tokenizer=True,
    ):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer
        )
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_embeddings(self, sentences, batch_size=32):
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings_list = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = self._pool_embeddings(outputs, inputs)
            embeddings_list.append(embeddings.cpu())

        return torch.cat(embeddings_list, dim=0).numpy()

    def _pool_embeddings(self, outputs, inputs):
        if self.pooling_method == "cls":
            return outputs.last_hidden_state[:, 0, :]
        elif self.pooling_method == "mean":
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_method == "pooler_output" and hasattr(
            outputs, "pooler_output"
        ):
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state.mean(dim=1)


# Data processing
def create_text_column(df):
    df["text"] = df.apply(
        lambda x: " ".join([f"{col}:{val}" for col, val in x.items()]), axis=1
    )
    return df


# TSNE embedding and plotting
def apply_tsne(embedding_matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(embedding_matrix)


def create_tsne_dataframe(tsne_results, patient_data):
    tsne_df = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    for col, values in patient_data.items():
        tsne_df[col] = values
    return tsne_df


app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "maxWidth": "1600px",
        "margin": "0 auto",
        "padding": "20px",
        "backgroundColor": "#F5F5F5",
    },
    children=[
        # Header Section
        html.Div(
            style={
                "backgroundColor": "#333333",
                "color": "white",
                "padding": "20px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "borderRadius": "10px",
                "marginBottom": "30px",
            },
            children=[
                # Logo on the left
                html.Img(
                    src="https://raw.githubusercontent.com/lab-rasool/HoneyBee/refs/heads/main/docs/assets/images/HoneyBee.png",
                    style={
                        "width": "120px",
                        "height": "auto",
                    },
                ),
                # Title and subtitle in the center
                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.H1(
                            "Welcome to HoneyBee Visualization Tool",
                            style={"margin": "0"},
                        ),
                        html.P(
                            "Your clinical data embedding and visualization assistant"
                        ),
                    ],
                ),
                # Upload button on the right
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "borderColor": "#888888",
                        "textAlign": "center",
                        "backgroundColor": "#FFFFFF",
                        "cursor": "pointer",
                        "transition": "border 0.3s ease",
                        "color": "#000000",
                    },
                    multiple=False,
                ),
            ],
        ),
        # Main Content
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "marginBottom": "30px",
            },
            children=[
                html.Button(
                    "Download Embeddings CSV",
                    id="download-button",
                    style={
                        "padding": "10px 20px",
                        "fontSize": "16px",
                        "backgroundColor": "#333333",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "marginTop": "20px",
                    },
                ),
                dcc.Download(id="download-csv"),
            ],
        ),
        html.Div(
            id="output-data-upload",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        html.Hr(),
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": "20px",
            },
            children=[
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": "Gatortron Base", "value": "UFNLP/gatortron-base"},
                        {"label": "BERT Base", "value": "bert-base-uncased"},
                        {
                            "label": "ClinicalBERT",
                            "value": "emilyalsentzer/Bio_ClinicalBERT",
                        },
                    ],
                    placeholder="Select Model for Embedding",
                    value="bert-base-uncased",  # Default value
                    style={"width": "100%", "padding": "10px"},
                ),
                dcc.Dropdown(
                    id="color-dropdown",
                    placeholder="Select Variable for Color Coding",
                    style={"width": "100%", "padding": "10px"},
                ),
            ],
        ),
        dcc.Loading(
            id="loading",
            type="circle",
            color="#FF6347",  # Tomato color spinner
            children=[
                dcc.Graph(id="tsne-scatter"),
            ],
        ),
        html.Hr(),
        html.H3(
            "Uploaded Data Preview",
            style={
                "textAlign": "center",
                "color": "#333333",
                "marginBottom": "10px",
            },
        ),
        dash_table.DataTable(
            id="data-table",
            page_size=10,
            style_table={
                "overflowX": "auto",
                "margin": "0 auto",
                "width": "100%",
            },
            style_cell={
                "textAlign": "center",
                "padding": "10px",
                "fontSize": "14px",
            },
            style_header={
                "backgroundColor": "#333333",
                "color": "white",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#f9f9f9",
                },
                {
                    "if": {"state": "active"},
                    "backgroundColor": "#D3D3D3",
                    "border": "1px solid #FF6347",
                },
            ],
        ),
        # Footer Section
        html.Div(
            style={
                "backgroundColor": "#333333",
                "color": "white",
                "padding": "10px",
                "textAlign": "center",
                "borderRadius": "10px",
                "marginTop": "30px",
            },
            children=[
                html.P("© 2024 HoneyBee Visualization Tool. All Rights Reserved."),
                html.A(
                    "Privacy Policy",
                    href="#",
                    style={"color": "#FF6347", "marginRight": "15px"},
                ),
                html.A(
                    "Terms of Service",
                    href="#",
                    style={"color": "#FF6347", "marginLeft": "15px"},
                ),
            ],
        ),
    ],
)


# Modify load_and_process_data to save embeddings into the DataFrame
@app.callback(
    [
        Output("output-data-upload", "children"),
        Output("color-dropdown", "options"),
        Output("color-dropdown", "value"),
        Output("data-table", "data"),
        Output("data-table", "columns"),
    ],
    [Input("upload-data", "contents"), Input("model-dropdown", "value")],
)
def load_and_process_data(contents, model_name):
    global global_tsne_df, global_df_with_embeddings
    if contents is None:
        return html.Div("Please upload a CSV file."), [], None, [], []

    # Parse the uploaded data
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(StringIO(decoded.decode("utf-8")))
    df = create_text_column(df)

    # Embed the data
    embedder = HuggingFaceEmbedder(
        model_name=model_name, pooling_method="mean", max_length=512
    )

    embeddings_list = []
    for index, row in df.iterrows():
        embedding = embedder.generate_embeddings(row["text"])
        embeddings_list.append(embedding.flatten())

    # Convert embeddings to DataFrame
    embeddings_df = pd.DataFrame(embeddings_list)
    embeddings_df.columns = [f"embedding_{i}" for i in range(embeddings_df.shape[1])]

    # Append embeddings to the original DataFrame
    global_df_with_embeddings = pd.concat([df, embeddings_df], axis=1)

    # Apply TSNE
    embedding_matrix = embeddings_df.to_numpy()
    tsne_results = apply_tsne(embedding_matrix)
    global_tsne_df = create_tsne_dataframe(
        tsne_results, {col: df[col].tolist() for col in df.columns}
    )

    # Update dropdown options
    dropdown_options = [
        {"label": col, "value": col}
        for col in global_tsne_df.columns
        if col not in ["tsne_1", "tsne_2"]
    ]
    color_variable = dropdown_options[0]["value"] if dropdown_options else None

    # Prepare data table columns and data
    table_columns = [{"name": i, "id": i} for i in global_df_with_embeddings.columns]
    table_data = global_df_with_embeddings.to_dict("records")

    return (
        html.Div("Data uploaded successfully"),
        dropdown_options,
        color_variable,
        table_data,
        table_columns,
    )


@app.callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    global global_df_with_embeddings
    if global_df_with_embeddings is None:
        return None

    # Save the DataFrame to a temporary file
    filename = "df_with_embeddings.csv"
    global_df_with_embeddings.to_csv(filename, index=False)

    # Serve the file for download
    return send_file(filename)


@app.callback(Output("tsne-scatter", "figure"), Input("color-dropdown", "value"))
def update_plot(color_variable):
    if global_tsne_df is None or color_variable is None:
        return {}

    # Generate plot
    fig = px.scatter(
        global_tsne_df,
        x="tsne_1",
        y="tsne_2",
        color=color_variable,
        title=f"t-SNE Plot of Embeddings - Color Coded by {color_variable}",
        labels={"tsne_1": "t-SNE Component 1", "tsne_2": "t-SNE Component 2"},
        hover_data={col: True for col in global_tsne_df.columns},
    )
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
