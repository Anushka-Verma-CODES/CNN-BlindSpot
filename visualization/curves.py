import plotly.graph_objects as go

def plot_loss_curve(history):

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = epochs,
        y = history['train_loss'],
        mode = 'lines',
        name = 'Training Loss',
        line = dict(color = 'blue')
    ))


    fig.add_trace(go.Scatter(
        x = epochs,
        y = history['val_loss'],
        mode = 'lines',
        name = 'Validation Loss',
        line = dict(color = 'Red')
    ))


    fig.update_layout(
        title = "Loss Curve",
        xaxis_title = "Epochs",
        yaxis_title = "Loss",
        template = "plotly_white"
    )

    return fig



def plot_accuracy_curve(history):

    epochs = list(range(1, len(history['train_acc']) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = epochs,
        y = history['train_acc'],
        mode = 'lines',
        name = 'Training Acc',
        line = dict(color = 'blue')
    ))

    fig.add_trace(go.Scatter(
        x = epochs,
        y = history['val_acc'],
        mode = 'lines',
        name = 'Validation Acc',
        line = dict(color = 'red')
    ))

    fig.update_layout(
        title = "Accuracy Curve",
        xaxis_title = "Epochs",
        yaxis_title = "Accuracy",
        template = "plotly_white"
    )

    return fig



if __name__ == "__main__":
    
    dummy_history = {
        'train_loss': [0.9, 0.6, 0.4, 0.2, 0.1],
        'val_loss': [0.9, 0.65, 0.5, 0.6, 0.8],
        'train_acc': [0.5, 0.6, 0.7, 0.8, 0.95],
        'val_acc': [0.5, 0.55, 0.6, 0.58, 0.55]
    }


    loss_fig = plot_loss_curve(dummy_history)
    print("Loss Figure created ...")
    print("Traces generated: ", len(loss_fig.data))