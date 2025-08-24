# 🐶 Dog Breed Classifier

A deep learning project that identifies dog breeds using **pre-trained CNN models (ResNet, AlexNet, VGG)** in Python.
Built as part of **Udacity’s AI Programming with Python Nanodegree**, this project demonstrates transfer learning, data preprocessing, and results evaluation for image classification tasks.

---

## 🚀 Features

* 📂 **Image Classification** → Classifies dog images into breeds using pre-trained CNNs.
* 🧠 **Transfer Learning** → Leverages ResNet, AlexNet, and VGG architectures.
* ⚡ **Custom Dataset Handling** → Works with both provided dog image dataset and uploaded test images.
* 📊 **Evaluation Metrics** → Reports classification accuracy and results statistics.
* 🛠️ **Modular Codebase** → Includes reusable scripts for training, prediction, and evaluation.

---

## 📂 Project Structure

<pre>
dog-breed-classifier/
│
├── adjust_results4_isadog.py              # Adjusts results to check if classified labels are dogs
├── alexnet_pet-images.txt                 # Classification results using AlexNet on pet_images
├── alexnet_uploaded-images.txt            # Classification results using AlexNet on uploaded_images
├── calculates_results_stats.py            # Calculates overall classification statistics
├── check_images.py                        # Main pipeline to run classification and compare results
├── check_images.txt                       # Example output file from check_images.py
├── classifier.py                          # Contains the image classifier function (PyTorch + pretrained models)
├── classify_images.py                     # Classifies images with chosen CNN model
├── dognames.txt                           # List of valid dog names for classification reference
├── get_input_args.py                      # Parses command-line arguments (model, dir, dogfile)
├── get_pet_labels.py                      # Extracts pet labels from image filenames
├── imagenet1000_clsid_to_human.txt        # Mapping of ImageNet class IDs to human-readable labels
├── print_functions_for_lab_checks.py      # Helper functions for lab testing/debugging
├── print_results.py                       # Prints final results summary in a readable format
├── resnet_pet-images.txt                  # Classification results using ResNet on pet_images
├── resnet_uploaded-images.txt             # Classification results using ResNet on uploaded_images
├── run_models_batch.sh                    # Bash script to run model training/testing on pet_images
├── run_models_batch_uploaded.sh           # Bash script to run model training/testing on uploaded_images
├── test_classifier.py                     # Unit test for classifier function
├── vgg_pet-images.txt                     # Classification results using VGG on pet_images
├── vgg_uploaded-images.txt                # Classification results using VGG on uploaded_images
└── README.md                              # Project documentation

</pre>

---

## 🖥️ How It Works

1. Loads a dataset of dog images from `pet_images/`.
2. Extracts true labels from filenames (e.g., "golden\_retriever\_01.jpg" → "golden retriever").
3. Uses a **pre-trained CNN** (ResNet, AlexNet, or VGG) from ImageNet to predict image labels.
4. Compares predicted labels with true labels and determines if the prediction is a dog.
5. Calculates and prints accuracy statistics (overall accuracy, dog vs. not-dog, breed match).

---

## ⚙️ How It Works

The project runs an **end-to-end image classification pipeline** using pre-trained deep learning models (AlexNet, ResNet, VGG).

<pre>
📸 Input Image
       │
       ▼
🧩 Pre-trained CNN Model (AlexNet / ResNet / VGG)
       │
       ▼
🏷️ Predicted Label
       │
       ▼
🐶 Check if Label is a Dog (using dognames.txt)
       │
       ▼
📊 Calculate Accuracy & Stats
       │
       ▼
🖨️ Print Final Results
</pre>

👉 Users can test the classifier with either **provided datasets (`pet_images/`)** or their own **uploaded images (`uploaded_images/`)**, and compare performance across CNN architectures.

---

## ⚡ Usage

### Requirements

* Python 3.6+
* PyTorch
* NumPy

### Example Run

```bash
# Run classifier with ResNet model
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt
```
---

## 📊 Results

During experimentation, three pre-trained CNN architectures (**AlexNet**, **ResNet**, and **VGG**) were tested on the dataset.

| Model   | % Correct Dogs | % Correct Not-a-Dog | % Correct Breed | Match Labels |
| ------- | -------------- | ------------------- | --------------- | ------------------- |
| ResNet | 100%           | 90%                | 90.0%           | 82.5%            |
| AlexNet  | 100%           | 100%               | 80.0%           | 75.0%                |
| VGG     | 100%           | 100%                | 93.3%           | 87.5%            |

### 📌 Key Insights

* **VGG** achieved **100% accuracy** for both "dogs" and "not-a-dog" classification, and the **highest breed classification accuracy (93.3%)**, making it the best overall model.
* **ResNet** performed better than AlexNet in dog breed classification but fell short in correctly identifying "not-a-dog" images.
* **AlexNet** was the weakest at breed classification, though it matched VGG in detecting dogs vs. not-a-dogs.
---

## 🎯 Why This Project?

This project demonstrates the power of **transfer learning** by applying ImageNet-trained CNNs to real-world classification tasks.
It highlights key skills in:

* Deep learning with PyTorch
* Image preprocessing & evaluation
* Modular Python scripting for AI workflows

---

## 👩‍💻 Author

**Bisan Abu Zubaida**
AI Programming with Python Nanodegree – Udacity
Passionate about Deep Learning, Python, and AI projects.

📂 Portfolio: [GitHub](https://github.com/Bisan-Abuzubaida)

---
