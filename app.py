# Complete MNIST Digit Recognition - Streamlit Web App
# Train models and test predictions all in one web interface

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
from PIL import Image
import cv2
import pandas as pd
import io
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MNISTProject:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data_variants = {}
        
    def load_and_preprocess_data(self):
        """Load MNIST data and create different preprocessing versions"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Create different input representations
        self.data_variants = {
            'Raw (0-255)': {
                'x_train': x_train.reshape(-1, 28, 28, 1).astype('float32'),
                'x_test': x_test.reshape(-1, 28, 28, 1).astype('float32'),
                'y_train': keras.utils.to_categorical(y_train, 10),
                'y_test': keras.utils.to_categorical(y_test, 10),
                'description': 'Raw pixel values (0-255)'
            },
            'Normalized (0-1)': {
                'x_train': x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0,
                'x_test': x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0,
                'y_train': keras.utils.to_categorical(y_train, 10),
                'y_test': keras.utils.to_categorical(y_test, 10),
                'description': 'Normalized pixel values (0-1)'
            },
            'MLP Flattened': {
                'x_train': x_train.reshape(-1, 784).astype('float32') / 255.0,
                'x_test': x_test.reshape(-1, 784).astype('float32') / 255.0,
                'y_train': keras.utils.to_categorical(y_train, 10),
                'y_test': keras.utils.to_categorical(y_test, 10),
                'description': 'Flattened for MLP (784 features)'
            },
            'Noisy Data': {
                'x_train': None,
                'x_test': None,
                'y_train': keras.utils.to_categorical(y_train, 10),
                'y_test': keras.utils.to_categorical(y_test, 10),
                'description': 'Normalized + Gaussian noise'
            }
        }
        
        # Add noisy data
        x_train_noisy = x_train.astype('float32') / 255.0
        x_test_noisy = x_test.astype('float32') / 255.0
        noise_train = np.random.normal(0, 0.1, x_train_noisy.shape)
        noise_test = np.random.normal(0, 0.1, x_test_noisy.shape)
        x_train_noisy = np.clip(x_train_noisy + noise_train, 0, 1)
        x_test_noisy = np.clip(x_test_noisy + noise_test, 0, 1)
        
        self.data_variants['Noisy Data']['x_train'] = x_train_noisy.reshape(-1, 28, 28, 1)
        self.data_variants['Noisy Data']['x_test'] = x_test_noisy.reshape(-1, 28, 28, 1)
        
        return self.data_variants
    
    def create_cnn_model(self, input_shape):
        """Create a simple CNN model"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        return model
    
    def create_mlp_model(self, input_shape):
        """Create a simple MLP model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        return model
    
    def train_model(self, variant_name, data, progress_bar, status_text):
        """Train a single model variant"""
        status_text.text(f"Training on {variant_name}...")
        
        # Choose model architecture
        if variant_name == 'MLP Flattened':
            model = self.create_mlp_model((784,))
        else:
            model = self.create_cnn_model((28, 28, 1))
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train model (reduced epochs for web interface)
        history = model.fit(
            data['x_train'], data['y_train'],
            batch_size=128,
            epochs=3,  # Reduced for faster training in web app
            validation_split=0.1,
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(data['x_test'], data['y_test'], verbose=0)
        
        # Store results
        self.models[variant_name] = model
        self.results[variant_name] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'history': history,
            'description': data['description']
        }
        
        return test_acc

def preprocess_image_for_prediction(image_array):
    """Preprocess uploaded image for prediction"""
    # Resize to 28x28
    img_resized = cv2.resize(image_array, (28, 28))
    
    # Convert to grayscale if needed
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Invert if needed (MNIST has white digits on black background)
    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized
    
    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Reshape for model
    img_final = img_normalized.reshape(1, 28, 28, 1)
    
    return img_final, img_resized

def plot_results_comparison(results):
    """Create results comparison chart"""
    variants = list(results.keys())
    accuracies = [results[v]['accuracy'] for v in variants]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(variants, accuracies, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
    
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, (variant, acc) in enumerate(zip(variants, accuracies)):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Highlight best model
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('gold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig

def plot_sample_predictions(model, test_data, test_labels, title="Sample Predictions"):
    """Show sample predictions"""
    num_samples = 10
    predictions = model.predict(test_data[:num_samples], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        if len(test_data.shape) == 4:
            img = test_data[i].reshape(28, 28)
        else:
            img = test_data[i].reshape(28, 28)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {test_labels[i]}\nPred: {predicted_labels[i]}', 
                         color='green' if test_labels[i] == predicted_labels[i] else 'red')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🔢", layout="wide")
    
    st.title("🔢 MNIST Handwritten Digit Recognition")
    st.markdown("**Complete Training & Testing Pipeline in One Web App**")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Choose Section:", 
                           ["🏠 Home", "🎯 Train Models", "🧪 Test Predictions", "📊 Results Analysis"])
    
    if page == "🏠 Home":
        st.header("🎯 Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 What This Project Does")
            st.markdown("""
            This project demonstrates how **different input preprocessing methods** 
            affect machine learning model performance on handwritten digit recognition.
            
            **4 Different Preprocessing Methods:**
            - **Raw Data**: Original pixel values (0-255)
            - **Normalized**: Scaled to 0-1 range  
            - **MLP Flattened**: 1D vectors for neural networks
            - **Noisy Data**: Added Gaussian noise for robustness testing
            """)
        
        with col2:
            st.subheader("🚀 How to Use")
            st.markdown("""
            1. **Train Models**: Go to the training section to train all 4 models
            2. **View Results**: See performance comparisons and analysis
            3. **Test Predictions**: Upload your own digit images for testing
            4. **Compare Performance**: Understand which preprocessing works best
            """)
        
        st.markdown("---")
        st.info("👈 Use the sidebar to navigate between sections. Start with **Train Models** to begin!")
    
    elif page == "🎯 Train Models":
        st.header("🎯 Model Training Dashboard")
        
        if st.button("🚀 Start Training All Models", type="primary"):
            project = MNISTProject()
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Loading MNIST dataset..."):
                status_text.text("📥 Loading MNIST dataset...")
                data_variants = project.load_and_preprocess_data()
                st.success("✅ Dataset loaded successfully!")
            
            # Training progress
            st.subheader("🔄 Training Progress")
            results_placeholder = st.empty()
            
            # Train each model
            for i, (variant_name, data) in enumerate(data_variants.items()):
                progress = (i + 1) / len(data_variants)
                progress_bar.progress(progress)
                
                accuracy = project.train_model(variant_name, data, progress_bar, status_text)
                
                # Update results table
                current_results = {k: v for k, v in project.results.items()}
                results_df = pd.DataFrame.from_dict(current_results, orient='index')
                results_df = results_df[['accuracy', 'description']].round(4)
                results_placeholder.dataframe(results_df)
            
            status_text.text("✅ Training completed!")
            progress_bar.progress(1.0)
            
            # Store results in session state
            st.session_state['trained_models'] = project
            st.session_state['training_completed'] = True
            
            st.success("🎉 All models trained successfully! Go to 'Results Analysis' to see detailed comparisons.")
        
        # Show previous results if available
        if 'training_completed' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Latest Training Results")
            project = st.session_state['trained_models']
            results_df = pd.DataFrame.from_dict(project.results, orient='index')
            results_df = results_df[['accuracy', 'description']].round(4)
            st.dataframe(results_df)
    
    elif page == "📊 Results Analysis":
        if 'training_completed' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Train Models' section!")
            return
        
        project = st.session_state['trained_models']
        st.header("📊 Detailed Results Analysis")
        
        # Performance comparison
        st.subheader("🏆 Model Performance Comparison")
        fig_comparison = plot_results_comparison(project.results)
        st.pyplot(fig_comparison)
        
        # Results table
        st.subheader("📋 Detailed Results Table")
        results_df = pd.DataFrame.from_dict(project.results, orient='index')
        results_df = results_df[['accuracy', 'loss', 'description']].round(4)
        results_df = results_df.sort_values('accuracy', ascending=False)
        st.dataframe(results_df, use_container_width=True)
        
        # Best model analysis
        best_model_name = max(project.results.keys(), key=lambda k: project.results[k]['accuracy'])
        best_model = project.models[best_model_name]
        
        st.subheader(f"🥇 Best Model: {best_model_name}")
        st.metric("Best Accuracy", f"{project.results[best_model_name]['accuracy']:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix for best model
            st.subheader("🎯 Confusion Matrix (Best Model)")
            variant_data = project.data_variants[best_model_name]
            test_data = variant_data['x_test']
            test_labels = np.argmax(variant_data['y_test'], axis=1)
            
            predictions = best_model.predict(test_data, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            
            fig_cm = plot_confusion_matrix(test_labels, predicted_labels, 
                                         f"Confusion Matrix - {best_model_name}")
            st.pyplot(fig_cm)
        
        with col2:
            # Sample predictions
            st.subheader("🔍 Sample Predictions")
            fig_samples = plot_sample_predictions(best_model, test_data, test_labels,
                                                f"Sample Predictions - {best_model_name}")
            st.pyplot(fig_samples)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        results_list = [(name, result['accuracy']) for name, result in project.results.items()]
        results_list.sort(key=lambda x: x[1], reverse=True)
        
        st.markdown(f"""
        **Performance Ranking:**
        1. 🥇 **{results_list[0][0]}**: {results_list[0][1]:.4f} accuracy
        2. 🥈 **{results_list[1][0]}**: {results_list[1][1]:.4f} accuracy  
        3. 🥉 **{results_list[2][0]}**: {results_list[2][1]:.4f} accuracy
        4. 4️⃣ **{results_list[3][0]}**: {results_list[3][1]:.4f} accuracy
        
        **What We Learned:**
        - **Normalization** typically improves model performance significantly
        - **CNN architectures** outperform MLPs on image data
        - **Noise** reduces accuracy but can improve model robustness
        - **Preprocessing choice** has a major impact on final performance
        """)
    
    elif page == "🧪 Test Predictions":
        st.header("🧪 Test Your Own Digits")
        
        # Load best model if available
        if 'training_completed' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Train Models' section!")
            return
        
        project = st.session_state['trained_models']
        best_model_name = max(project.results.keys(), key=lambda k: project.results[k]['accuracy'])
        best_model = project.models[best_model_name]
        
        st.success(f"✅ Using best model: **{best_model_name}** (Accuracy: {project.results[best_model_name]['accuracy']:.4f})")
        
        # File uploader
        uploaded_file = st.file_uploader("📁 Upload a digit image (0-9)", 
                                       type=['png', 'jpg', 'jpeg'],
                                       help="Best results with clear, single digits on light background")
        
        if uploaded_file is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📷 Original Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", width=200)
            
            with col2:
                st.subheader("⚙️ Processed Image")
                img_array = np.array(image)
                processed_img, display_img = preprocess_image_for_prediction(img_array)
                st.image(display_img, caption="Processed (28x28)", width=200, clamp=True)
            
            with col3:
                st.subheader("🎯 Prediction Results")
                
                # Make prediction
                prediction = best_model.predict(processed_img, verbose=0)
                predicted_digit = np.argmax(prediction[0])
                confidence = prediction[0][predicted_digit] * 100
                
                # Show prediction
                st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                st.markdown(f"**Confidence: {confidence:.1f}%**")
                
                # Confidence meter
                st.progress(int(confidence))           # as int (0–100 range)

            
            # Detailed probabilities
            st.subheader("📊 All Digit Probabilities")
            prob_data = pd.DataFrame({
                'Digit': range(10),
                'Probability (%)': prediction[0] * 100
            }).round(2)
            
            # Highlight predicted digit
            prob_data['Predicted'] = prob_data['Digit'] == predicted_digit
            
            st.dataframe(prob_data, hide_index=True, use_container_width=True)
        
        # Instructions
        st.markdown("---")
        st.subheader("📝 How to Get Best Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **✅ Good Images:**
            - Single digit (0-9)
            - Dark digit on light background
            - Centered and clear
            - Similar to handwritten style
            """)
        
        with col2:
            st.markdown("""
            **❌ Avoid:**
            - Multiple digits in one image
            - Very small or very large digits
            - Blurry or unclear images
            - Complex backgrounds
            """)

if __name__ == "__main__":
    main()