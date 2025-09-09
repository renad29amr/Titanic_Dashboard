import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dash import Dash,html,dcc,Input,Output
import plotly.express as px

df = pd.read_csv('Titanic.csv')
df.head()
df.describe()
df.info()
df.isnull().sum().sort_values(ascending=False)
df.drop(columns=["Cabin"], inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="Set2")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()
sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x="Pclass", y="Fare", data=df, palette="pastel")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df, palette="coolwarm")
plt.title("Age vs Fare by Survival Status")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
matrix = df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr()
sns.heatmap(matrix, annot=True, cmap='Reds', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
app = Dash()

app.layout = html.Div([
    html.H1("Titanic Dashboard,",style={'color':'red'}),
    html.Label("Select type of Graph"),
    dcc.Dropdown(
        options=['Bar', 'Histogram', 'Scatter', 'Heatmap', 'Box'],
        id='plot-type',
        value='bar'
    ),
    dcc.Graph(id='graph')
])

@app.callback(
    Output('graph', 'figure'),
    Input('plot-type', 'value')
)
def update_graph(plot_type):
    if plot_type == 'Bar':
        fig = px.bar(df, x="Sex", color="Survived", barmode="group",
                     title="Survival Count by Gender")
    elif plot_type == 'Histogram':
        fig = px.histogram(df, x="Age", nbins=20, color="Survived",
                           title="Age Distribution")
    elif plot_type == 'Scatter':
        fig = px.scatter(df, x="Age", y="Fare", color="Survived",
                         title="Age vs Fare")
    elif plot_type == 'Heatmap':
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    elif plot_type == 'Box':
        fig = px.box(df, x="Pclass", y="Fare", color="Pclass",
                     title="Fare by Passenger Class")
    else:
        fig = px.scatter(df, x="Age", y="Fare", title="Default Scatter Plot")
    return fig

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=10000, debug=False)


