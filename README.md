# COVID-19 Chest X-ray Image Classification Using a Spiking Neural Network

This project presents a replication of the architecture proposed in *Spiking Neural Network Classification of X-ray Chest Images* (2025), implemented using the PyTorchSpiking framework. The goal is to classify COVID-19 cases from chest X-ray images while evaluating the model’s energy efficiency.

---

## 📂 Project Structure

- `env.yml`: Conda environment specification containing all necessary dependencies.
- `main.py`: Main script for training the spiking neural network (SNN) model.
- `test.ipynb`: Notebook for performing inference and measuring energy consumption using CodeCarbon after training is complete.

---

## ⚙️ Setup Instructions

### 1️⃣ Create the Environment

To set up the Python environment with all required dependencies:

```bash
conda env create -f env.yml
conda activate covid-snn
```

> Note: Replace `covid-snn` with your desired environment name if you wish to customize it.

---

### 2️⃣ Train the Model

To train the model from scratch:

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Train the convolutional spiking neural network for 20 epochs
- Save logs and model performance metrics

---

### 3️⃣ Run Inference and Measure Energy

After training, open the notebook to perform inference and estimate energy:

```bash
jupyter notebook test.ipynb
```

The notebook includes:
- Model loading and evaluation on the test set
- ROC AUC and confusion matrix display
- Energy consumption estimation using the CodeCarbon framework

> Ensure `main.py` has completed successfully before running `test.ipynb`, as it assumes the trained model is available.

---

## 🔋 Energy Efficiency

The model's inference energy usage is approximated using the [CodeCarbon](https://github.com/mlco2/codecarbon) library. While this approach does not provide per-inference granularity (like neuromorphic profiling on Intel Loihi), it offers a practical, system-level estimate when simulating SNNs on CPU/GPU.

---

## 📦 Dependencies

All dependencies are listed in `env.yml`, including:

- `pytorch`
- `pytorch-spiking`
- `codecarbon`
- `numpy`, `matplotlib`, `scikit-learn`, `jupyter`
---

## 🧪 Dataset

This project uses the [Extensive and Augmented COVID-19 X-ray and CT Chest Images Dataset](https://doi.org/10.17632/8h65ywd2jr.2).

Ensure the dataset is available in the expected directory or update the path accordingly in `main.py`.

---

## 🔄 Future Improvements

- Incorporate multi-step spike encoding for richer temporal representation
- Perform systematic hyperparameter tuning
- Run multiple training trials to compute confidence intervals
- Compare with simpler models (e.g., shallow CNN)
- Benchmark on real neuromorphic hardware for precise energy metrics

---

## 📜 License

This code is released for academic and research purposes only.

---

## 🙏 Acknowledgments

- Original paper by Marco Gatti et al., *Knowledge-Based Systems*
- PyTorchSpiking library authors
- CodeCarbon energy tracking framework



