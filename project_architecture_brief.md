# Project Architecture Overview
## LiDAR-Camera Fusion Robustness in Adverse Weather

---

## **The Problem**

Autonomous driving detectors fail 20-40% in rain/fog/snow, especially at oblique angles (90°, 180° azimuth). The degradation is angle-dependent and not well understood.

---

## **Solution**

### **Phase 1**

Using pre-trained 3D object detection models to measure how detection accuracy varies by:
- **Angle**: 0°, 45°, 90°, 180° (azimuth)
- **Weather**: Clear, rain, fog, night
- **Datasets**: nuScenes (primary) + Ithaca365 (validation) + CADC (snow)

---

## **Combined Pre-trained Models**

### **Model 1: SAMFusion good for bad conditions**

**What it is**: Deep learning model designed specifically for adverse weather (ECCV 2024)

**Strengths**:
- Explicitly trained on rain/fog/snow scenarios
- Sensor-adaptive: learns which sensor to trust when
- Pre-trained weights available (Princeton CI Lab)
- Robust in bad weather

---

### **Model 2: BEVFusion Accuracy**

Industry-standard 3D object detection model (ICRA 2023)

**Strengths**:
- production-proven
- High accuracy in clear weather
- Pre-trained weights available (NVIDIA NGC)

---

## **Strategy**


Run **both models**, combine their predictions:

```
Input: LiDAR + Camera
    ↓
    ├→ SAMFusion: "Car at [10m, 2m], confidence 92%"
    │
    ├→ BEVFusion: "Car at [10.1m, 1.9m], confidence 88%"
    │
    └→ ENSEMBLE: "Car at [10.05m, 1.95m], confidence 90%"
```

### **Why This Works**

| Scenario | SAMFusion Alone | BEVFusion Alone | Ensemble |
|----------|-----------------|-----------------|----------|
| **Clear weather** | Good | Excellent | Excellent |
| **Heavy fog + 90° angle** | Excellent | Poor | Excellent |
| **Rain + 45° angle** | Excellent | Good | Excellent |


---



### **Method: Weighted by Weather**
- Bad weather → 70% SAM, 30% BEV
- Good weather → 50% SAM, 50% BEV

### **Method: Confidence **
Weight by each model's own confidence score

---

## **Phase 1**

**Run both models**, create three heatmaps:

```
Heatmap 1: SAMFusion alone (angle × weather → mAP)
Heatmap 2: BEVFusion alone (angle × weather → mAP)
Heatmap 3: COMBINED (angle × weather → mAP)

Question: Does Combined models outperform both in adverse weather + high angles?
```

---

### **Model Weights**

1. **SAMFusion pre-trained weights**
   - Pre-trained on: Adverse weather scenarios

2. **BEVFusion pre-trained weights**
   - Pre-trained on: nuScenes clear weather

### **Datasets for Inference**

3. **Test datasets**
   - nuScenes 
   - Ithaca365 
   - CADC 

### **Analysis**

4. **Combined processing**
   - Run SAMFusion on each frame
   - Run BEVFusion on each frame
   - Combine predictions 
   - Bin by angle + weather
   - Calculate ensemble mAP per bin

---


## **Deliverables**

**Phase 1**
- Systematic characterization using the **combined** approach
- Three heatmaps: SAMFusion alone, BEVFusion alone, Ensemble combined
- Cross-dataset validation
- Statistical analysis: "Does combined models outperform both in adverse weather and clear conditions?"
- answer: "how here's how combined approach handles different angles"

**Phase 2**
- Optimization of ensemble weighting (if time allows)
- Real-world deployment considerations

---


**Combining Models**:
- SAMFusion: Weather robustness specialist
- BEVFusion: Accuracy specialist

**Research question**: "Can Combined fusion outperform single models at extreme angle-weather combinations?"

---

## **Complete Tech Stack**

### **Programming Languages**
- **Python 3.10+**

### **ML Frameworks**
- **PyTorch 2.0+**
- **MMDetection3D** 
- **ONNX** 

### **Dataset Tools**
- **nuScenes DevKit** 
- **KITTI Tools** 
- **Open3D** 
- **CARMAKER** 

### **Data Processing & Analysis**
- **Pandas** 
- **NumPy/SciPy** 
- **scikit-learn** 

### **Visualization**
- **Matplotlib** 
- **Plotly** 
- **Jupyter** 

### **Development Tools**
- **VS Code** 
- **Git** 

### **Pre-trained Models**
- **SAMFusion** (ECCV 2024) - Weather-robust multi-modal fusion
- **BEVFusion** (ICRA 2023) - Accuracy-optimized baseline

---
## **CANBUS**

### **Option 1: Input Only**
Read vehicle speed, steering angle, acceleration to provide context for detection.
- Improves object classification by understanding vehicle motion

### **Option 2: Output Only**
Send brake commands to trigger AEB based on detection confidence.
- Detection confidence > threshold → activate AEB via CANBUS

### **Option 3: Full AEB Loop (Read + Write)**
Read vehicle state, then decide AEB trigger based on detection + speed + angle + weather.
- Read: speed, brake status, steering from CANBUS
- Detect: objects, confidence, angle, weather conditions
- Decide: trigger AEB threshold based on all factors
- Write: brake command via CANBUS

### **Option 4: Validation & Sanity Check**
Use CANBUS data to validate detection logic.
- braking → detected objects should be stable ahead
- accelerating → detected objects should move backward

### **Option 5: Contextual Weighting**
Use vehicle speed to dynamically adjust SAMFusion vs BEVFusion weights.
- Low speed/bad weather → favor SAMFusion (robustness)
- High speed/clear weather → favor BEVFusion (accuracy)
