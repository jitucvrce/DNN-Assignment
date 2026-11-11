import streamlit as tf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from engine import generate_adding_problem, generate_mock_har, UnifiedSequenceModel

# Page configuration profiles
tf.set_page_config(page_title="RNN Optimization Dashboard", layout="wide", initial_sidebar_state="expanded")

tf.title("🌐 Deep Recurrent Network Benchmarking Portal")
tf.markdown("Evaluate information bottlenecks, vanishing gradients, and sequence tracking criteria in real time.")

# Sidebar Controls
tf.sidebar.header("🛠️ Global Experiment Parameters")
experiment_task = tf.sidebar.selectbox("Select Target Benchmark Suite", ["Task 4: Adding Problem Matrix", "Task 3: HAR Classification Data"])
selected_model_type = tf.sidebar.selectbox("Choose Cell Architecture", ["RNN", "LSTM", "GRU"])
hidden_dimension_size = tf.sidebar.slider("Hidden Dimension Size (h)", 16, 128, 64, step=16)
training_epochs = tf.sidebar.slider("Epoch Count Limit", 5, 50, 15)
learning_rate = tf.sidebar.number_input("Optimizer Learning Rate", value=1e-3, format="%.4f")

if experiment_task == "Task 4: Adding Problem Matrix":
    sequence_length_t = tf.sidebar.select_slider("Sequence Temporal Length (T)", options=[50, 100, 200, 400], value=100)
    ablation_setting = tf.sidebar.radio("LSTM Forget Gate Init Bias Ablation", [1.0, 0.0], index=0)
else:
    sequence_length_t = 128
    ablation_setting = 1.0

# Initializing trigger mechanisms
if tf.sidebar.button("🚀 Run Live Architecture Benchmarks"):
    tf.subheader(f"📊 Live Stream Metrics: {selected_model_type} running on {experiment_task}")
    
    # Progress feedback hooks
    progress_bar = tf.progress(0.0)
    metric_status_block = tf.empty()
    chart_render_block = tf.empty()
    
    # Dataset pre-allocation
    if experiment_task == "Task 4: Adding Problem Matrix":
        X_train, Y_train = generate_adding_problem(800, sequence_length_t)
        X_test, Y_test = generate_adding_problem(200, sequence_length_t)
        criterion_loss = nn.MSELoss()
        output_classes = 1
    else:
        # Mock HAR logic path setup
        X_raw, y_raw, _ = generate_mock_har(800, 128, 9)
        X_train = torch.tensor(X_raw[:600], dtype=torch.float32)
        Y_train = torch.tensor(y_raw[:600], dtype=torch.long)
        X_test = torch.tensor(X_raw[600:], dtype=torch.float32)
        Y_test = torch.tensor(y_raw[600:], dtype=torch.long)
        criterion_loss = nn.CrossEntropyLoss()
        output_classes = 6

    # Instantiate structural system networks
    input_features_count = 2 if experiment_task == "Task 4: Adding Problem Matrix" else 9
    model = UnifiedSequenceModel(selected_model_type, input_features_count, hidden_dimension_size, output_classes, ablation_setting)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    time_history = []
    
    # Execute optimization pipeline sequence loop
    for epoch in range(1, training_epochs + 1):
        start_time = time.time()
        model.train()
        
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion_loss(predictions, Y_train if output_classes == 1 else Y_train)
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start_time
        loss_history.append(loss.item())
        time_history.append(elapsed)
        
        # Incremental runtime visualization metrics
        progress_bar.progress(epoch / training_epochs)
        metric_status_block.metric(
            label=f"Epoch {epoch}/{training_epochs} Evolution Floor", 
            value=f"{loss.item():.5f}", 
            delta=f"{elapsed:.3f} seconds/epoch"
        )
        
        # Real-time loss tracking charts
        tracking_df = pd.DataFrame({"Optimization Trajectory Loss": loss_history})
        chart_render_block.line_chart(tracking_df)
        time.sleep(0.05)

    # Post-run testing evaluation matrix metrics
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss_val = criterion_loss(test_predictions, Y_test).item()
        
    tf.success("🎉 Experiment completed successfully!")
    
    # Multi-column analytical breakdown dashboards
    col1, col2, col3 = tf.columns(3)
    col1.metric("Final Test Evaluation Floor", f"{test_loss_val:.5f}")
    col2.metric("Mean Latency Target Time", f"{np.mean(time_history):.3f}s")
    
    if experiment_task == "Task 4: Adding Problem Matrix":
        is_solved = "SUCCESS (Solved)" if test_loss_val < 0.05 else "FAILED (Stalled)"
        col3.metric("Baseline Threshold Convergence Status", is_solved)
    else:
        predicted_classes = torch.argmax(test_predictions, dim=1)
        accuracy = (predicted_classes == Y_test).float().mean().item() * 100
        col3.metric("Overall Validation Target Accuracy", f"{accuracy:.2f}%")

else:
    # Default initial state placeholder layout
    tf.info("Adjust the parameters in the left sidebar configuration panel and click 'Run Live Architecture Benchmarks' to launch.")
    
    # Conceptual structural overview blocks
    with tf.expander("ℹ️ Theoretical Reference Architecture Manual"):
        tf.markdown("""
        ### Parameters and Layer Memory Boundaries
        * **Vanilla RNN Parameter Counts:** $4(hd + h^2 + h)$ — Prone to severe exponential decay curves under sequence horizons where $T \ge 100$.
        * **GRU Target Layouts:** $3 \times (2(hd + h^2 + h))$ — Combines reset and update vectors to protect historical features across medium-range sequence horizons.
        * **LSTM Core Amortization Layouts:** $4 \times (2(hd + h^2 + h))$ — Isolates memory registers from processing loops to scale effectively across long sequences ($T \ge 400$).
        """)
