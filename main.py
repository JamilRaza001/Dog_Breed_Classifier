import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import io

# Configure Streamlit page
st.set_page_config(
    page_title="🐕 Elite Dog Breed Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; }
    .performance-badge { font-size: 1.2rem; font-weight: bold; text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem;
        border-radius: 25px; margin: 0.5rem 0; }
    .prediction-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem;
        border-radius: 15px; color: white; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        border: 2px solid #ffd700; }
    .top5-card { padding: 0.6rem; border-radius: 8px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        background: linear-gradient(180deg,#ffffff,#f7fbff); min-height: 110px; display:flex; flex-direction:column;
        justify-content:center; align-items:center; }
    .top5-breed { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.25rem; color:#111; }
    .top5-rank { font-weight: 700; color: #333; margin-bottom: 0.25rem; }
    /* Center the "Top 5 Predictions" header */
    .top5-header { text-align: center; margin-top: 1rem; margin-bottom: 0.5rem; font-weight:700; font-size:1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- DOG BREEDS & METRICS ---
DOG_BREEDS = [
    'Afghan',
    'African Wild Dog',
    'Airedale',
    'American Hairless',
    'American Spaniel',
    'Basenji',
    'Basset',
    'Beagle',
    'Bearded Collie',
    'Bermaise',
    'Bichon Frise',
    'Blenheim',
    'Bloodhound',
    'Bluetick',
    'Border Collie',
    'Borzoi',
    'Boston Terrier',
    'Boxer',
    'Bull Mastiff',
    'Bull Terrier',
    'Bulldog',
    'Cairn',
    'Chihuahua',
    'Chinese Crested',
    'Chow',
    'Clumber',
    'Cockapoo',
    'Cocker',
    'Collie',
    'Corgi',
    'Coyote',
    'Dalmation',
    'Dhole',
    'Dingo',
    'Doberman',
    'Elk Hound',
    'French Bulldog',
    'German Sheperd',
    'Golden Retriever',
    'Great Dane',
    'Great Perenees',
    'Greyhound',
    'Groenendael',
    'Irish Spaniel',
    'Irish Wolfhound',
    'Japanese Spaniel',
    'Komondor',
    'Labradoodle',
    'Labrador',
    'Lhasa',
    'Malinois',
    'Maltese',
    'Mex Hairless',
    'Newfoundland',
    'Pekinese',
    'Pit Bull',
    'Pomeranian',
    'Poodle',
    'Pug',
    'Rhodesian',
    'Rottweiler',
    'Saint Bernard',
    'Schnauzer',
    'Scotch Terrier',
    'Shar_Pei',
    'Shiba Inu',
    'Shih-Tzu',
    'Siberian Husky',
    'Vizsla',
    'Yorkie'
]

ACTUAL_METRICS = {
    'top1_accuracy': 90.57,
    'top5_accuracy': 99.71,
    'top3_accuracy': 98.71,
    'total_test_samples': 700,
    'num_classes': 70
}

# --- Model loading and VGG extractor caching ---
@st.cache_resource
def load_trained_model(path="improved_70_class_model.keras"):
    try:
        model = load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_vgg_feature_extractor():
    try:
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    except Exception as e:
        st.warning("Could not load VGG16 ImageNet weights; using random weights. Error: {}".format(e))
        vgg = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    vgg.trainable = False
    return vgg

# --- Image preprocessing ---
def preprocess_image(uploaded_image):
    try:
        if hasattr(uploaded_image, "read"):
            uploaded_image.seek(0)
            image_bytes = uploaded_image.read()
            img = Image.open(io.BytesIO(image_bytes))
        elif isinstance(uploaded_image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(uploaded_image))
        elif isinstance(uploaded_image, Image.Image):
            img = uploaded_image
        elif isinstance(uploaded_image, str):
            img = Image.open(uploaded_image)
        else:
            try:
                image_bytes = uploaded_image.getvalue()
                img = Image.open(io.BytesIO(image_bytes))
            except Exception:
                raise ValueError("Unsupported image type")

        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

# --- Adaptive prediction ---
def predict_breed(model, img_array, top_k=5):
    try:
        try:
            model_input = model.input_shape
            if isinstance(model_input, list):
                model_input = model_input[0]
        except Exception:
            model_input = None

        expected = None
        if model_input is not None:
            expected = tuple(model_input[1:])

        if expected is not None and len(expected) == 3 and expected[2] == 3:
            preds = model.predict(img_array, verbose=0)
        else:
            vgg = get_vgg_feature_extractor()
            features = vgg.predict(img_array, verbose=0)
            if expected is not None and len(expected) == 3 and expected == features.shape[1:]:
                preds = model.predict(features, verbose=0)
            elif expected is not None and len(expected) == 1 and expected[0] == features.shape[-1]:
                gap = np.mean(features, axis=(1, 2))
                preds = model.predict(gap, verbose=0)
            else:
                flat = features.reshape((features.shape[0], -1))
                try:
                    preds = model.predict(flat, verbose=0)
                except Exception:
                    try:
                        gap = np.mean(features, axis=(1, 2))
                        preds = model.predict(gap, verbose=0)
                    except Exception:
                        preds = model.predict(img_array, verbose=0)

        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, 0)

        top_indices = np.argsort(preds[0])[::-1][:top_k]
        top_probabilities = preds[0][top_indices]

        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            breed_name = DOG_BREEDS[idx] if idx < len(DOG_BREEDS) else f"Breed_{idx}"
            results.append({
                'rank': i + 1,
                'breed': breed_name,
                'confidence': float(prob),
                'percentage': float(prob * 100)
            })
        return results
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return []

# --- Visual helpers ---
def create_confidence_chart(predictions):
    df = pd.DataFrame(predictions)
    # sort descending so chart shows highest on top
    df = df.sort_values(by='percentage', ascending=True)  # ascending True so largest at bottom => horizontal bars read top->bottom nicely
    fig = px.bar(
        df,
        x='percentage',
        y='breed',
        orientation='h',
        title='🎯 Top Breed Predictions (Confidence %)',
        labels={'percentage': 'Confidence (%)', 'breed': ''},
        color='percentage',
        color_continuous_scale='RdYlGn',
        text='percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=360, margin=dict(l=120, r=20, t=40, b=40), yaxis={'categoryorder':'total ascending'}, showlegend=False, title_x=0.5)
    return fig

def get_prediction_confidence_level(confidence_pct):
    if confidence_pct >= 80:
        return "🌟 EXCELLENT", "confidence-excellent"
    elif confidence_pct >= 60:
        return "✅ VERY GOOD", "confidence-good"
    elif confidence_pct >= 40:
        return "👍 GOOD", "confidence-good"
    else:
        return "🤔 UNCERTAIN", "confidence-fair"

def create_elite_performance_metrics():
    st.markdown("""
    <div class="achievement-banner">
        🏆 ELITE PERFORMANCE ACHIEVED! 🏆<br>
        This model ranks among the top-performing dog breed classifiers worldwide!
    </div>
    """, unsafe_allow_html=True)
    st.subheader("🎯 Actual Test Performance (700 Test Images)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-container elite-metric"><h2>90.57%</h2><p><strong>Top-1 Accuracy</strong></p><small>634/700 correct predictions</small></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-container elite-metric"><h2>98.71%</h2><p><strong>Top-3 Accuracy</strong></p><small>691/700 in top-3</small></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-container elite-metric"><h2>99.71%</h2><p><strong>Top-5 Accuracy</strong></p><small>698/700 in top-5</small></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-container"><h2>70</h2><p><strong>Dog Breeds</strong></p><small>Multi-class classification</small></div>""", unsafe_allow_html=True)

# --- Main app ---
def main():
    st.markdown('<h1 class="main-header">🏆 Elite Dog Breed Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<div class="performance-badge">🌟 Elite Performance: 90.57% Accuracy • 99.71% Top-5 • 70 Breeds 🌟</div>', unsafe_allow_html=True)

    model, error = load_trained_model()
    if model is None:
        st.error(f"❌ Failed to load model: {error}")
        st.info("Please make sure 'improved_70_class_model.keras' is in the same directory as this script.")
        st.stop()

    with st.sidebar:
        st.header("🏆 Elite Model")
        st.markdown(f"""
        ### 🎯 **Verified Performance**
        - ✅ **Top-1**: {ACTUAL_METRICS['top1_accuracy']}%
        - ✅ **Top-3**: {ACTUAL_METRICS['top3_accuracy']}%  
        - ✅ **Top-5**: {ACTUAL_METRICS['top5_accuracy']}%
        - 🧠 **Architecture**: VGG16 + Elite Classifier
        - 📊 **Test Dataset**: {ACTUAL_METRICS['total_test_samples']} images
        """)
        st.success("🏆 **ELITE STATUS ACHIEVED!**\nThis model performs at expert-level accuracy!")
        st.subheader("🔧 Prediction Settings")
        show_top_k = st.slider("Show top predictions", 3, 10, 5)
        show_confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 5)

    tab1, tab2, tab3 = st.tabs(["🔍 Elite Predictions", "🏆 Performance Metrics", "🏗️ Architecture"])

    with tab1:
        st.subheader("Upload Dog Images for Elite-Level Breed Prediction")
        
        # Multiple image processing options
        col_upload1, col_upload2 = st.columns([2, 1])
        with col_upload1:
            uploaded_files = st.file_uploader(
                "Choose dog images for elite-level analysis...",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload multiple dog images to experience 90.57% accuracy predictions! Supports batch processing."
            )
        
        with col_upload2:
            if uploaded_files:
                st.info(f"📁 **{len(uploaded_files)} images** selected for analysis")
                batch_analysis = st.checkbox("Enable Batch Summary", value=True, help="Show summary statistics for all images")
                show_individual = st.checkbox("Show Individual Analysis", value=True, help="Show detailed analysis for each image")
            else:
                st.info("👆 Select multiple images above")
                batch_analysis = False
                show_individual = True

        if uploaded_files:
            # Batch processing summary
            if batch_analysis and len(uploaded_files) > 1:
                st.markdown("## 📊 Batch Processing Summary")
                
                # Process all images for batch summary
                batch_results = []
                batch_progress = st.progress(0)
                batch_status = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    batch_status.text(f"Processing {file.name}... ({idx + 1}/{len(uploaded_files)})")
                    img_array, _ = preprocess_image(file)
                    if img_array is not None:
                        predictions = predict_breed(model, img_array, top_k=5)
                        if predictions:
                            batch_results.append({
                                'filename': file.name,
                                'top_breed': predictions[0]['breed'],
                                'confidence': predictions[0]['percentage'],
                                'predictions': predictions
                            })
                    batch_progress.progress((idx + 1) / len(uploaded_files))
                
                batch_status.empty()
                batch_progress.empty()
                
                if batch_results:
                    # Create batch summary
                    st.markdown("### 🏆 Batch Results Overview")
                    
                    # Summary metrics
                    avg_confidence = np.mean([r['confidence'] for r in batch_results])
                    high_conf_count = len([r for r in batch_results if r['confidence'] >= 80])
                    unique_breeds = len(set([r['top_breed'] for r in batch_results]))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Images Processed", len(batch_results))
                    with col2:
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    with col3:
                        st.metric("High Confidence (≥80%)", high_conf_count)
                    with col4:
                        st.metric("Unique Breeds Found", unique_breeds)
                    
                    # Batch results table
                    batch_df = pd.DataFrame([
                        {
                            'Image': r['filename'],
                            'Predicted Breed': r['top_breed'],
                            'Confidence': f"{r['confidence']:.1f}%",
                            'Assessment': get_prediction_confidence_level(r['confidence'])[0]
                        }
                        for r in batch_results
                    ])
                    st.dataframe(batch_df, use_container_width=True, hide_index=True)
                    
                    # Breed distribution chart
                    if len(batch_results) > 1:
                        breed_counts = pd.DataFrame([r['top_breed'] for r in batch_results], columns=['breed']).value_counts('breed').reset_index()
                        breed_counts.columns = ['Breed', 'Count']
                        fig_breeds = px.bar(breed_counts, x='Breed', y='Count', 
                                          title='🐕 Predicted Breed Distribution Across All Images',
                                          color='Count', color_continuous_scale='viridis')
                        fig_breeds.update_layout(height=400, title_x=0.5, xaxis_tickangle=-45)
                        st.plotly_chart(fig_breeds, use_container_width=True, key="batch_breed_distribution")
                    
                    # Confidence distribution chart
                    if len(batch_results) > 1:
                        conf_data = pd.DataFrame([
                            {'Image': r['filename'], 'Confidence': r['confidence']}
                            for r in batch_results
                        ])
                        fig_conf = px.histogram(conf_data, x='Confidence', nbins=10,
                                              title='📈 Confidence Score Distribution',
                                              labels={'Confidence': 'Confidence (%)', 'count': 'Number of Images'})
                        fig_conf.update_layout(height=300, title_x=0.5)
                        st.plotly_chart(fig_conf, use_container_width=True, key="batch_confidence_distribution")
                
                st.divider()
            
            # Individual image analysis
            if show_individual:
                st.markdown("## 🔍 Individual Image Analysis")
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.expander(f"📸 Image {i+1}: {uploaded_file.name}", expanded=len(uploaded_files) <= 3):
                        # two-column layout: left=image, right=top1 & short info
                        col1, col2 = st.columns([1, 1])

                        predictions = []

                        with col1:
                            try:
                                image_pil = Image.open(uploaded_file)
                                st.image(image_pil, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not display image: {e}")
                                image_pil = None

                        with col2:
                            with st.spinner("🔄 Elite AI analyzing image..."):
                                img_array, processed_img = preprocess_image(uploaded_file)
                                if img_array is not None:
                                    predictions = predict_breed(model, img_array, top_k=show_top_k)
                                    if predictions and len(predictions) > 0:
                                        # Keep Top-1 display inside the right column
                                        top_prediction = predictions[0]
                                        confidence_level, css_class = get_prediction_confidence_level(top_prediction['percentage'])
                                        st.markdown(f"""
                                            <div class="prediction-box {css_class}">
                                                <h2>🏆 Elite Prediction</h2>
                                                <h1>{top_prediction['breed']}</h1>
                                                <h2>{top_prediction['percentage']:.1f}% Confidence</h2>
                                                <h3>{confidence_level}</h3>
                                            </div>
                                            """, unsafe_allow_html=True)

                                        confidence = top_prediction['percentage']
                                        if confidence > 80:
                                            st.success("🌟 **ELITE CONFIDENCE**: Extremely reliable prediction.")
                                        elif confidence > 60:
                                            st.success("🎯 **HIGH CONFIDENCE**: Very reliable prediction.")
                                        elif confidence > 40:
                                            st.warning("✅ **GOOD CONFIDENCE**: Solid prediction, check alternatives.")
                                        else:
                                            st.info("🤔 **UNCERTAIN**: Consider top 3–5 predictions.")

                                    else:
                                        st.warning("No predictions returned by model.")
                                else:
                                    st.error("❌ Failed to preprocess the image. Please try a different image.")

                        # ---------- CENTERED FULL-WIDTH TOP-5 BLOCK ----------
                        if predictions is not None and len(predictions) > 0:
                            # Ensure we always show top-5 slots (fill with placeholders if needed)
                            top5 = predictions[:5]
                            # If fewer than 5 results, pad with blanks
                            while len(top5) < 5:
                                top5.append({'rank': len(top5)+1, 'breed': '—', 'confidence': 0.0, 'percentage': 0.0})

                            # Centered header
                            st.markdown("<div class='top5-header'>🔝 Top 5 Predictions</div>", unsafe_allow_html=True)

                            # Full width row of 5 columns so cards span the page and are centered
                            card_cols = st.columns(5)
                            for col, p in zip(card_cols, top5):
                                with col:
                                    st.markdown(
                                        f"""
                                        <div class="top5-card">
                                          <div class="top5-rank">Rank #{p['rank']}</div>
                                          <div class="top5-breed">{p['breed']}</div>
                                          <div style="font-weight:600">{p['percentage']:.1f}%</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                            # Single Top-5 chart (full width) with unique key
                            fig_top5 = create_confidence_chart([p for p in top5 if p['breed'] != '—'])
                            st.plotly_chart(fig_top5, use_container_width=True, key=f"top5_chart_image_{i}")

                            # Top-5 dataframe / quick table
                            df_top5 = pd.DataFrame([p for p in top5 if p['breed'] != '—'])
                            if not df_top5.empty:
                                df_top5['confidence'] = df_top5['percentage'].apply(lambda x: f"{x:.1f}%")
                                df_top5['assessment'] = df_top5['percentage'].apply(lambda x: get_prediction_confidence_level(x)[0])
                                st.dataframe(df_top5[['rank', 'breed', 'confidence', 'assessment']], use_container_width=True, hide_index=True)

                        # Detailed predictions (filtered by threshold)
                        if predictions is not None and len(predictions) > 0:
                            st.subheader("📊 Elite Model Detailed Analysis (Filtered by Confidence Threshold)")
                            filtered_predictions = [p for p in predictions if p['percentage'] >= show_confidence_threshold]
                            if filtered_predictions:
                                fig = create_confidence_chart(filtered_predictions)
                                st.plotly_chart(fig, use_container_width=True, key=f"detailed_chart_image_{i}")

                                df_predictions = pd.DataFrame(filtered_predictions)
                                df_predictions['confidence'] = df_predictions['percentage'].apply(lambda x: f"{x:.1f}%")
                                df_predictions['assessment'] = df_predictions['percentage'].apply(lambda x: get_prediction_confidence_level(x)[0])
                                st.dataframe(df_predictions[['rank', 'breed', 'confidence', 'assessment']], use_container_width=True, hide_index=True)
                            else:
                                st.warning(f"No predictions above {show_confidence_threshold}% confidence threshold.")
            else:
                st.info("💡 **Individual analysis disabled** - Only batch summary is shown above. Enable 'Show Individual Analysis' to see detailed results for each image.")
        else:
            st.markdown("""
            ### 🏆 Experience Elite-Level Dog Breed Classification with Multiple Images!

            Upload your dog images above to experience our **elite model** that achieved:
            - **90.57% accuracy** on test data (634/700 correct predictions)
            - **99.71% top-5 accuracy** (correct breed in top 5 for 698/700 images)
            - **98.71% top-3 accuracy** for practical applications
            
            #### 🚀 **New Multiple Image Features:**
            - **Batch Processing**: Upload multiple images at once
            - **Batch Summary**: Get overview statistics across all images
            - **Breed Distribution**: See which breeds appear most frequently
            - **Confidence Analysis**: Analyze prediction confidence patterns
            - **Individual Analysis**: Toggle detailed analysis per image
            """)
            st.subheader("🐕 70 Supported Dog Breeds")
            breeds_df = pd.DataFrame({
                'Dog Breeds': DOG_BREEDS[:35],
                'More Breeds': DOG_BREEDS[35:70] if len(DOG_BREEDS) > 35 else [''] * (35 - len(DOG_BREEDS[35:]))
            })
            st.dataframe(breeds_df, use_container_width=True, hide_index=True)

    with tab2:
        create_elite_performance_metrics()
        st.subheader("🧪 Actual Test Results Breakdown")
        test_results = {
            'Metric': ['Correct Top-1 Predictions', 'Correct in Top-3', 'Correct in Top-5', 'Total Test Images'],
            'Count': [634, 691, 698, 700],
            'Percentage': [90.57, 98.71, 99.71, 100],
            'Performance Level': ['Elite', 'Near Perfect', 'Virtually Perfect', 'Complete Dataset']
        }
        df_results = pd.DataFrame(test_results)
        fig_results = px.bar(df_results.iloc[:-1], x='Metric', y='Count', title='🎯 Test Results: Elite Performance Breakdown', text='Count', color='Percentage', color_continuous_scale='RdYlGn')
        fig_results.update_traces(texttemplate='%{text} / 700', textposition='outside')
        fig_results.update_layout(height=400, title_x=0.5)
        st.plotly_chart(fig_results, use_container_width=True, key="performance_metrics_chart")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🏗️ Architecture")
        st.markdown("Model architecture and training details go here (kept compact).")

if __name__ == "__main__":
    main()