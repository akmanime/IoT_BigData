import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Rental Guardian", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(r"Dataset\smart_rental_guardian_dataset.csv")

df = load_data()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        # Try joblib first (more reliable for sklearn models)
        import joblib
        return joblib.load("model_rf_building_fault.pkl")
    except Exception as e1:
        try:
            # Try pickle with different protocols
            with open("model_rf_building_fault.pkl", "rb") as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e2:
            try:
                with open("model_rf_building_fault.pkl", "rb") as f:
                    return pickle.load(f, encoding='bytes')
            except Exception as e3:
                # Silently return None - we'll handle this in the UI
                return None

model = load_model()

# ---------------- TITLE ----------------
st.title("🏠 Smart Rental Guardian Dashboard")
st.write("Dataset Visualization + AI Fault Detection")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Dashboard Controls")

# Get numeric columns only for sensor selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_sensor = st.sidebar.selectbox("Choose sensor to visualize:", numeric_cols)

# Fix: Only use numeric columns for threshold slider
if selected_sensor in numeric_cols:
    threshold = st.sidebar.slider(
        "Alert Threshold", 
        float(df[selected_sensor].min()), 
        float(df[selected_sensor].max()), 
        float(df[selected_sensor].mean())
    )
else:
    threshold = 0.0

date_filter = st.sidebar.checkbox("Use date filter")

# ---------------- DATE FILTER ----------------
if "timestamp" in df.columns and date_filter:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df = df.dropna(subset=["timestamp"])
    start = st.sidebar.date_input("Start date", df["timestamp"].min())
    end = st.sidebar.date_input("End date", df["timestamp"].max())
    df = df[(df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))]

# ---------------- LATEST DATA ----------------
latest = df.iloc[-1]
st.subheader("🔎 Real-Time Indicators")

# Show only numeric columns in metrics
display_cols = [col for col in numeric_cols if col != 'timestamp'][:6]
cols = st.columns(len(display_cols))
for i, col_name in enumerate(display_cols):
    try:
        value = latest[col_name]
        if pd.notna(value):
            cols[i].metric(col_name, f"{float(value):.2f}")
        else:
            cols[i].metric(col_name, "N/A")
    except Exception as e:
        cols[i].metric(col_name, "Error")

# ---------------- AI PREDICTION ----------------
st.subheader("🤖 AI Model Prediction")

if model is not None:
    try:
        # Prepare features - drop timestamp and any non-numeric columns
        features = df.select_dtypes(include=[np.number]).drop(columns=["timestamp"], errors="ignore")
        
        # Check if we have target column to exclude
        if 'target' in features.columns or 'label' in features.columns or 'fault' in features.columns:
            features = features.drop(columns=['target', 'label', 'fault'], errors='ignore')
        
        pred = model.predict([features.iloc[-1]])[0]
        if pred == 1:
            st.error("🚨 Risk / Fault Detected by AI Model")
        else:
            st.success("✅ System Operating Normally")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
else:
    st.info("ℹ️ Model file could not be loaded. The dashboard will work without AI predictions.")
    with st.expander("📝 How to fix this"):
        st.markdown("""
        **To enable AI predictions, retrain and save your model:**
        
        ```python
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        # Train your model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Save with joblib (recommended)
        joblib.dump(model, 'model_rf_building_fault.pkl')
        ```
        """)

# ---------------- ALERT SYSTEM ----------------
st.subheader("🚨 Sensor Alerts")
if latest[selected_sensor] > threshold:
    st.warning(f"{selected_sensor.upper()} exceeded alert threshold!")
else:
    st.info(f"{selected_sensor.upper()} is within normal range.")

# ---------------- GRAPH ----------------
st.subheader(f"📈 {selected_sensor} Over Time")
if "timestamp" in df.columns:
    fig = px.line(df, x="timestamp", y=selected_sensor, title=selected_sensor)
else:
    fig = px.line(df, y=selected_sensor, title=selected_sensor)
st.plotly_chart(fig, use_container_width=True)

# ---------------- STATUS PANEL ----------------
st.subheader("📊 System Health")
health = {
    "Gas": "🟢 Normal" if latest.get("gaz_ppm", 0) < 600 else "🔴 Critical",
    "Temperature": "🟢 Normal" if latest.get("temperature", 0) < 45 else "🔴 High",
    "Water Leak": "🟢 OK" if latest.get("fuite_eau", 0) == 0 else "🔴 Leak",
    "Fire": "🟢 OK" if latest.get("flamme", 0) == 0 else "🔴 Fire",
    "Voltage": "🟢 Normal" if 210 <= latest.get("tension", 0) <= 240 else "🔴 Fault"
}
st.table(pd.DataFrame.from_dict(health, orient="index", columns=["Status"]))

# ---------------- TEST NEW VALUES ----------------
st.subheader("🧪 Test de Nouvelles Valeurs")

col1, col2 = st.columns([2, 1])

with col1:
    st.write("**Entrez des valeurs personnalisées pour tester le système:**")
    
    # Create input fields for each sensor
    test_cols = st.columns(3)
    test_values = {}
    
    # Get all numeric columns except timestamp
    sensor_cols = [col for col in numeric_cols if col not in ['timestamp', 'target', 'label', 'fault']]
    
    for idx, sensor in enumerate(sensor_cols):
        col_idx = idx % 3
        with test_cols[col_idx]:
            # Get default value from latest data
            default_val = float(latest.get(sensor, df[sensor].mean()))
            min_val = float(df[sensor].min())
            max_val = float(df[sensor].max())
            
            test_values[sensor] = st.number_input(
                f"{sensor}",
                min_value=min_val * 0.5,  # Allow 50% below min
                max_value=max_val * 1.5,  # Allow 50% above max
                value=default_val,
                step=(max_val - min_val) / 100,
                key=f"test_{sensor}"
            )

with col2:
    st.write("**Actions:**")
    
    if st.button("🔄 Réinitialiser aux valeurs actuelles", use_container_width=True):
        st.rerun()
    
    if st.button("📊 Utiliser les valeurs moyennes", use_container_width=True):
        st.rerun()

# Display test results
st.write("---")
col_result1, col_result2 = st.columns(2)

with col_result1:
    st.write("**🔍 Analyse des Valeurs de Test:**")
    
    # Create comparison table
    comparison_data = []
    for sensor in sensor_cols:
        current = latest.get(sensor, 0)
        test = test_values[sensor]
        diff = test - current
        diff_percent = (diff / current * 100) if current != 0 else 0
        
        comparison_data.append({
            "Capteur": sensor,
            "Actuel": f"{current:.2f}",
            "Test": f"{test:.2f}",
            "Différence": f"{diff:+.2f} ({diff_percent:+.1f}%)"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

with col_result2:
    st.write("**🤖 Prédiction IA sur les Valeurs de Test:**")
    
    if model is not None:
        try:
            # Create DataFrame from test values
            test_df = pd.DataFrame([test_values])
            
            # Make prediction
            test_pred = model.predict(test_df)[0]
            
            # Display result
            if test_pred == 1:
                st.error("🚨 **DÉFAUT DÉTECTÉ**")
                st.write("Le modèle prédit un risque avec ces valeurs.")
            else:
                st.success("✅ **FONCTIONNEMENT NORMAL**")
                st.write("Le modèle prédit un fonctionnement normal.")
            
            # Show probabilities if available
            if hasattr(model, 'predict_proba'):
                test_proba = model.predict_proba(test_df)[0]
                
                st.write("**Probabilités:**")
                prob_col1, prob_col2 = st.columns(2)
                prob_col1.metric("Normal", f"{test_proba[0]:.1%}")
                prob_col2.metric("Défaut", f"{test_proba[1]:.1%}")
                
                # Progress bar
                st.progress(test_proba[1], text=f"Risque de défaut: {test_proba[1]:.1%}")
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
    else:
        st.warning("⚠️ Modèle non disponible")

# Health check for test values
st.write("**🏥 Vérification Santé des Valeurs de Test:**")

test_health = {}
alerts = []

# Check each sensor against thresholds
if "gaz_ppm" in test_values:
    if test_values["gaz_ppm"] < 600:
        test_health["Gas"] = "🟢 Normal"
    else:
        test_health["Gas"] = "🔴 Critical"
        alerts.append(f"⚠️ Gaz: {test_values['gaz_ppm']:.2f} ppm (seuil: 600)")

if "temperature" in test_values:
    if test_values["temperature"] < 45:
        test_health["Temperature"] = "🟢 Normal"
    else:
        test_health["Temperature"] = "🔴 High"
        alerts.append(f"⚠️ Température: {test_values['temperature']:.2f}°C (seuil: 45)")

if "fuite_eau" in test_values:
    if test_values["fuite_eau"] == 0:
        test_health["Water Leak"] = "🟢 OK"
    else:
        test_health["Water Leak"] = "🔴 Leak"
        alerts.append(f"⚠️ Fuite d'eau détectée: {test_values['fuite_eau']}")

if "flamme" in test_values:
    if test_values["flamme"] == 0:
        test_health["Fire"] = "🟢 OK"
    else:
        test_health["Fire"] = "🔴 Fire"
        alerts.append(f"⚠️ Flamme détectée: {test_values['flamme']}")

if "tension" in test_values:
    if 210 <= test_values["tension"] <= 240:
        test_health["Voltage"] = "🟢 Normal"
    else:
        test_health["Voltage"] = "🔴 Fault"
        alerts.append(f"⚠️ Tension anormale: {test_values['tension']:.2f}V (normale: 210-240V)")

# Display health status
health_col1, health_col2 = st.columns([1, 2])

with health_col1:
    st.table(pd.DataFrame.from_dict(test_health, orient="index", columns=["Status"]))

with health_col2:
    if alerts:
        st.warning("**Alertes détectées:**")
        for alert in alerts:
            st.write(alert)
    else:
        st.success("✅ Tous les paramètres sont dans les limites normales")

# ---------------- RAW DATA ----------------
with st.expander("📁 View Full Dataset"):
    st.dataframe(df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Smart Rental Guardian | AI Monitoring | 2025")