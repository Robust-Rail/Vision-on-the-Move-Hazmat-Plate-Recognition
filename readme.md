# UN number detection

## Business Understanding
For this project, we have been tasked with developing a machine learning model capable of recognizing UN number hazard plates. These plates, commonly displayed on freight train wagons, indicate the types of hazardous materials being transported. The successful implementation of this model will contribute to a more efficient and secure railway system across the EU.

<img src="images/hazard_plate.jpg" alt="Hazard Plate" width="500"/>

The hazard plates play a crucial role in ensuring the safety of transportation by providing essential information about the nature of the substances on board, such as flammability, toxicity, or corrosiveness. By automating the recognition process with machine learning, the handling and tracking of these hazardous materials can be streamlined, reducing manual labor and minimizing potential human errors.
 Determine business objectives

### Determine business objectives
#### Background
The specific expectations and objectives of the EU for this project are not yet fully defined, but the initiative's roots are clear. This project is spearheaded by the University of Twente, with researcher Mellisa Tijink serving as our supervisor. Our team, composed of pre-master's Computer Science students, has been tasked with developing the machine learning model. Mellisa Tijink plays a pivotal role as the intermediary between our team and major stakeholders, including ProRail, the EU, and other experts in the field.

This project is part of a broader initiative aimed at enhancing rail freight operations within Europe, aligning with the EU’s goals for improved efficiency and safety. More information on the initiative can be found on the official project site: [EU Rail FP5](https://projects.rail-research.europa.eu/eurail-fp5/).

Flagship Project 5: *TRANS4M-R aims to establish rail freight as the backbone of a low-emission, resilient European logistics chain that meets end-user needs. It focuses on two main technological clusters: 'Full Digital Freight Train Operation' and 'Seamless Freight Operation', which will develop and demonstrate solutions to increase rail capacity, efficiency, and cross-border coordination. By integrating Digital Automatic Coupler (DAC) solutions with software-defined systems, the project seeks to optimize network management and enhance cooperation among infrastructure managers. The ultimate goal is to create an EU-wide, interoperable rail freight framework with unified technologies and seamless operations across borders and various stakeholders, boosting the EU transport and logistics sector.*

#### Business objectives

**Primary Objective:** Develop an object detection model for UN number hazard plates on freight wagons.

**Sub-objectives:**
1. Detect and identify UN number hazard plates: Ensure the model can accurately locate hazard plates on freight wagons. 
2. Read and interpret the UN numbers: Implement recognition capabilities to accurately read the numbers on the detected plates.
3. Ensure model robustness and accuracy: Train the model to achieve high accuracy and reliability under various conditions (e.g., different lighting, weather).
4. Optimize model for speed: Make sure the model runs efficiently and in real-time to function on moving trains.
5. Adapt the model for moving environments: Design and test the model to handle the unique challenges of detecting and reading plates on trains in motion. 
### Assess Situation

#### Inventory of resources

**Business Experts:** Our team currently lacks extensive expertise in this area. We can consult Melissa for some questions, and we have an upcoming interview with a Swedish expert in the field of UN number hazard plates.

**Data Mining Team:** 
- Melissa Tijink (Researcher in Data Management & Biometrics/Electrical Engineering, Mathematics, and Computer Science)
- Ewaldo Nieuwenhuis (Pre-master student in Computer Science)
- Stanislav Levendeev (Pre-master student in Computer Science)

**Data:**
1. **Video Data of Freight Trains:** This consists of video footage of moving freight trains, where the freight wagons should display the UN numbers.
2. **Line Scan Camera Pictures:** These are high-resolution images of the train, but they are very spread out. It is still uncertain if these will be useful.
3. **Photos of ADR Warning Signs:** These are images of ADR signs on freight trains. However, this is not exactly what we need since our objective is to build a model that recognizes UN numbers.

**Computing Resources:** We have access to a cluster from the University of Twente, which we can use to train or fine-tune our model.

**Software:** We will use Python, Jupyter Notebook, Keras, PyTorch, and TensorFlow for analyzing, cleaning, preparing the data, and modeling. For data labeling, we will use [CVAT](https://www.cvat.ai/).
### Requirements, assumptions, and constraints

##### Requirements
- Object detection capability for UN number hazard plates.
- Text recognition to read and extract UN numbers.
- High accuracy and precision in detection and recognition.
- Robust performance under varying conditions (weather, lighting, speed).
- Speed optimization for fast processing with minimal lag
- Real-time processing for operation on moving trains.

##### Assumptions
- Consistent access to a high-performance computational cluster for model training and testing.
- The high-performance cluster is necessary due to the heavy processing demands of deep learning models.
- Local machines are not sufficient for the required high computational tasks.
- Project-specific data, including images and videos of freight trains with hazard plates, will be provided as planned.
- Data will include varied conditions (different lighting and weather) to ensure robustness.
- Access to diverse data is essential for creating a model that generalizes well to real-world scenarios.
- If the planned data is unavailable, additional time will be needed to source and prepare alternative public datasets.
- Sourcing alternative datasets may affect the project timeline and the quality of the final outcomes.
- The stakeholders will provide timely feedback to guide any changes or adaptations needed in the project.

##### Constraints
- The team has restricted experience with advanced object detection methods, which may impact the initial development and refinement of the model.
- Most of the available data is not labeled, presenting a challenge for training supervised machine learning models. Some labeled data exists but belongs to another researcher, and access to it is uncertain.
- The project must be completed within a short, 9-week period, which constrains the depth and breadth of potential research and model development.
- The dataset may be skewed with an overrepresentation of specific UN numbers from certain wagons, which could limit the model's ability to generalize across different scenarios.
- The size of the dataset makes it difficult to filter out specific wagons or relevant segments efficiently, posing a challenge for data processing and targeted training
#### Risks and Contingencies

**1. Lack of Data Access:**  
*Risk:* Currently, we do not have access to the necessary video or linescan data, and there is a risk that we may never obtain it.  
*Contingency Action:* Search for publicly available open-source datasets containing UN codes to proceed with model training and development.

**2. Loss of Access to the Computational Cluster:**  
*Risk:* While we currently have access to a high-performance cluster for training, loading, and fine-tuning models, there is a chance of losing this access due to technical failures or maintenance issues.  
*Contingency Action:* Prepare to train, load, and fine-tune a smaller version of the model locally on personal computers.

**3. Unavailability of Labeling Software:**  
*Risk:* We plan to label the data with the help of our supervisor, Melissa, which is essential for fine-tuning and evaluating the model. If this step is delayed or cannot occur, it will impede progress.  
*Contingency Action:* Learn how to use CVAT (Computer Vision Annotation Tool) and set it up on personal laptops to carry out data labeling independently.

**4. Inaccessibility of Personal Laptops:**  
*Risk:* Access to our laptops is crucial for development, data handling, and connecting to the cluster. If our laptops become unusable due to malfunction, our work will be disrupted.  
*Contingency Action:* Use backup laptops that are ready for project work to ensure continuity.
#### Terminology

**Business Terminology:**
- **UN Number Hazard Plates**: Identification plates with UN numbers that indicate the nature of hazardous materials, improving safety during transport.
- **Freight Trains**: Trains used for transporting goods, especially relevant when carrying hazardous materials.
- **Flagship Project 5**: A project within the European "Europe’s Rail" initiative, focused on applying technologies to enhance rail transport safety.
- **ADR (European Agreement concerning the International Carriage of Dangerous Goods by Road)**: International regulations governing the transport of hazardous goods.
- **ProRail**: The Dutch railway network manager responsible for maintaining the railways.
- **Line-Scan Camera**: A camera that captures images one line at a time for capturing objects like fast-moving trains.

**Data Mining Terminology:**
- **CRISP-DM (Cross-Industry Standard Process for Data Mining)**: A widely used methodology for managing data mining projects, consisting of six phases:
  - **Business Understanding**: Defining objectives from a business perspective.
  - **Data Understanding**: Collecting and analyzing data to gain insights.
  - **Data Preparation**: Preparing data, such as annotating and normalizing, for model training.
  - **Modeling**: Selecting and training models for the desired task.
  - **Evaluation**: Assessing model performance using specific metrics.
  - **Deployment**: Implementing the model in real-world applications.

- **Object Detection**: Identifying specific objects (e.g., hazard plates) within images or videos.
- **Optical Character Recognition (OCR)**: Extracting text from images, used here to read numbers on hazard plates.
- **YOLO (You Only Look Once)**: A fast object detection model ideal for real-time applications.
- **Faster R-CNN**: A more accurate but slightly slower object detection model, suitable for complex environments.
- **Annotation**: Marking data (e.g., video frames) with labels like bounding boxes to create ground truth for model training.
- **Bounding Boxes**: Rectangular boxes used in image processing to define regions of interest around an object.
- **Normalization**: Adjusting data to a standard scale to ensure consistency in model input.
- **Augmentation**: Enhancing training data through techniques like contrast adjustment to improve model robustness.
- **Average Precision (AP)**: A metric for evaluating the accuracy of object detection models.
- **Tesseract**: A commonly used OCR tool for extracting alphanumeric text from images.
- **HOG (Histogram of Oriented Gradients)**: A feature descriptor used in object detection, especially for detecting shapes or text.
- **Saliency Detection**: An algorithmic technique to identify key areas within images for focused analysis.
- **Support Vector Regression (SVR)**: A machine learning algorithm for regression tasks, sometimes used to create likelihood maps for image processing.

### Determine Data Mining Goals

#### Data Mining Goals
**Primary Data Mining Goal:** Create and train an object detection model capable of identifying and interpreting UN number hazard plates on freight wagons in real-time.

**Specific Data Mining Goals:**
1. **Object Detection and Localization**: Develop a model that achieves a high AP score for accurately detecting and localizing hazard plates on freight wagons within each video frame.

2. **OCR for UN Number Extraction:** Use Tesseract to apply Optical Character Recognition (OCR) for accurately reading UN numbers on detected plates, aiming to optimize precision and minimize errors in text recognition.

3. **Robustness Across Variable Conditions**: Enhance the model’s robustness by training it on datasets representing diverse lighting and weather conditions, with a goal to maintain high AP scores across these environments.

4. **Optimization for Real-Time Processing**: Implement real-time object detection and OCR capabilities to ensure the model operates at a frame rate suitable for analyzing images from moving trains.

#### Data Mining Success Criteria

- **Object Detection AP**: Achieve an Mean Average Precision (mAP) of at least 0.70 for detecting and localizing hazard plates across varied conditions.
- **OCR Precision for UN Numbers**: Ensure the Tesseract OCR module achieves high accuracy in reading UN numbers, even under challenging conditions, with a target precision score above 0.95.
- **Processing Speed**: Ensure the model achieves a processing time per frame under 100 milliseconds to maintain real-time functionality.
- **Environmental Robustness**: Maintain consistent mAP scores across different lighting and weather conditions.
