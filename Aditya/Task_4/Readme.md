# Neural Network for Learning Addition

This project implements a simple **feedforward neural network** from scratch in Python (using only `numpy`) to learn the operation of **integer addition** (e.g., `20 + 20 = 40`, `30 + 46 = 76`, etc.).

The goal is to **hard-code the forward and backward passes** without using deep learning frameworks like TensorFlow or PyTorch. All core operations (forward propagation, backpropagation, weight updates) are written manually.

---

## ✨ Features
- Minimal neural network: 2 inputs → 1 hidden layer → 1 output  
- Manual backpropagation (no autograd)  
- Mean Squared Error (MSE) loss  
- ReLU activation in the hidden layer  
- Gradient descent updates  
- Trained to approximate addition for numbers in `[0, 99]`  

---

## 📂 Project Structure
├── Task_4.py # Main Python script with model and training
├── README.md # Project documentation

---

## ⚙️ How It Works
1. **Dataset Generation**  
   - Inputs: all integer pairs `(a, b)` where `a, b ∈ [0, 99]`.  
   - Targets: the sum `a + b`.  
   - Example: `(30, 46) → 76`.  

2. **Model Architecture**  
   - Input layer: 2 neurons (for the two numbers).  
   - Hidden layer: 8 neurons, ReLU activation.  
   - Output layer: 1 neuron, linear activation.  

3. **Training**  
   - Loss: Mean Squared Error (MSE).  
   - Optimizer: Gradient Descent (manually implemented).  
   - Full-batch training across 2000 epochs.  

4. **Prediction**  
   - After training, the model can predict the sum of two unseen numbers. 

###🧠 Mathematical Background

#Forward pass:

z1 = X · W1 + b1
a1 = ReLU(z1)
z2 = a1 · W2 + b2
y_pred = z2

#Loss:

L = mean((y_true - y_pred)^2)

#Backpropagation:

dL/dW2 = a1ᵀ · dL/dy
dL/db2 = sum(dL/dy)
dL/dW1 = Xᵀ · (dL/dz1)
dL/db1 = sum(dL/dz1)
 

---

## 🚀 Usage

### 1. Clone repository
```bash
git clone https://github.com/your-username/addition-nn.git
cd addition-nn

