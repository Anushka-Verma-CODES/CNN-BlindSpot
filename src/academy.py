import streamlit as st
import numpy as np
import plotly.express as px

def render_knowledge_check(key, question, options, correct_idx, explanation):
    st.markdown("---")
    st.write(f"**🧠 Knowledge Check:** {question}")
    state_key = f"quiz_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    choice = st.radio("Select an answer:", options, index=None, key=f"radio_{key}")
    
    if st.button("Submit", key=f"btn_{key}"):
        st.session_state[state_key] = choice
        
    if st.session_state[state_key] is not None:
        if st.session_state[state_key] == options[correct_idx]:
            st.success("Correct!")
        else:
            st.error("Incorrect. Review the explanation below.")
        st.info(f"**Explanation:** {explanation}")

def chapter_1():
    st.header("Chapter 1: How a Computer Sees an Image")
    st.markdown("""
    ### 1.1 The Digital Image & Tensors
    When humans look at a photograph, our brains instantly process shapes, context, and depth. A computer, however, only sees a giant grid of numbers. 
    
    An image is constructed from thousands of tiny squares called **pixels**. In a standard color image, every single pixel is represented by three overlapping color channels: **Red, Green, and Blue (RGB)**. The intensity of each color is measured on a scale from 0 (completely dark) to 255 (maximum brightness).
    
    In Deep Learning, we store this data in multi-dimensional matrices called **Tensors**. If you have a 224x224 pixel color image, it is loaded into memory as a 3D tensor with the shape `(224, 224, 3)`. That is exactly 150,528 individual numbers that the neural network must process!
    """)
    
    st.markdown("#### Interactive Demo: RGB Pixels")
    st.write("Adjust the sliders to see how combining three numerical intensities creates the colors we see on screen.")
    c1, c2, c3 = st.columns(3)
    r = c1.slider("Red (0-255)", 0, 255, 255)
    g = c2.slider("Green (0-255)", 0, 255, 128)
    b = c3.slider("Blue (0-255)", 0, 255, 0)
    color_hex = f"#{r:02x}{g:02x}{b:02x}"
    st.markdown(f'<div style="background-color: {color_hex}; height: 80px; border-radius: 8px; border: 1px solid #555;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1.2 The Preprocessing Pipeline
    Neural networks are rigid mathematical functions. They cannot dynamically adjust to images of random sizes. Furthermore, feeding numbers as large as 255 into complex calculus equations causes a problem called "exploding gradients," where the math breaks down.
    
    Before an image reaches the neural network, it must pass through a strict **preprocessing pipeline**:
    1. **Resizing & Cropping:** The image is forced into a fixed square (e.g., 224x224 pixels).
    2. **Normalization:** The pixel values are mathematically scaled down from the `0 to 255` range into a smaller decimal range (usually between `-1.0` and `1.0`, or `0.0` and `1.0`). This keeps the network's calculations stable.
    """)
    render_knowledge_check("ch1_q1", "Why must we normalize pixel values before feeding them into a Neural Network?", 
        [
            "To remove the color channels and make the image grayscale.", 
            "To scale large numbers (0-255) down to small decimals to prevent the network's mathematical calculations from becoming unstable.", 
            "To artificially increase the resolution of the image."
        ], 1, 
        "Normalization keeps numerical values small and stable. Unnormalized inputs cause numerical overflow during deep matrix multiplications.")

def chapter_2():
    st.header("Chapter 2: What Is a Neural Network?")
    st.markdown("""
    ### 2.1 The Architecture of a Network
    A Neural Network is an algorithm loosely inspired by biological brains. It consists of layers of artificial **neurons** (nodes).
    * **Inputs:** The raw data (our image tensor) fed into the first layer.
    * **Weights:** The connections between neurons. You can think of weights as the network's *memory* or *learned rules*. They are just numbers. If a weight is high, the connection is strong.
    * **Activations:** Mathematical gates that decide if a neuron should "fire" and pass data to the next layer based on the sum of its inputs and weights.
    
    ### 2.2 Training vs. Inference
    * **Training (The Backward Pass):** The network starts completely ignorant. We feed it an image of a dog, and it guesses randomly. We calculate how wrong it was (the *Loss*), and use calculus (*Backpropagation*) to update the **weights** so it guesses slightly better next time. This requires millions of images and massive GPU compute power.
    * **Inference (The Forward Pass):** Once the network is highly accurate, we "freeze" its weights. It stops learning. We give it new images and it applies its frozen weights to make a prediction. This requires much less computing power.
    
    ### 2.3 Pretrained Models
    In this application, we use a **pretrained model**. We did not train it ourselves. We downloaded the frozen weights from PyTorch. Because we are only doing *inference*, the model runs easily on a standard CPU.
    """)
    render_knowledge_check("ch2_q1", "When you mask part of the image in the lab, is the network learning to recognize the grey box?", 
        [
            "Yes, the network's weights are updating to learn about occlusion.", 
            "No, the weights are frozen. We are only running inference to see how the already-trained model reacts."
        ], 1, 
        "Our backend strictly uses `model.eval()` and `torch.inference_mode()`. No training or weight updating occurs in this application.")

def chapter_3():
    st.header("Chapter 3: Understanding CNNs")
    st.markdown("""
    ### 3.1 Why standard Neural Networks fail on images
    If you feed a 2D image into a standard neural network, it flattens the image into one giant 1D line of numbers. It completely destroys the **spatial relationships**—it forgets that a pixel was next to another pixel!
    
    ### 3.2 Convolution: The Sliding Window
    A **Convolutional Neural Network (CNN)** preserves 2D space. It slides a small grid (a **kernel** or **filter**, usually 3x3 pixels) across the image. 
    
    As the filter slides, it performs **Element-wise Multiplication and Summation**. The filter is mathematically tuned to react to specific patterns (vertical lines, corners, color gradients). The resulting output map is called a **Feature Map**.
    """)
    
    st.markdown("#### Interactive Demo: The Convolution Operation")
    col1, col2, col3 = st.columns(3)
    with col1:
        img_pattern = st.selectbox("1. Choose Image Pattern:", ["Vertical Line", "Horizontal Line"])
        img = np.array([[0,0,10,0,0],[0,0,10,0,0],[0,0,10,0,0],[0,0,10,0,0],[0,0,10,0,0]]) if img_pattern == "Vertical Line" else np.array([[0,0,0,0,0],[0,0,0,0,0],[10,10,10,10,10],[0,0,0,0,0],[0,0,0,0,0]])
        fig1 = px.imshow(img, color_continuous_scale="gray", text_auto=True, title="Input (5x5)")
        fig1.update_coloraxes(showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        filter_type = st.selectbox("2. Choose Filter:", ["Vertical Edge", "Horizontal Edge"])
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) if filter_type == "Vertical Edge" else np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        fig2 = px.imshow(kernel, color_continuous_scale="RdBu", text_auto=True, title="Kernel (3x3)")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.write("3. Calculate Feature Map")
        if st.button("Apply Convolution ➡️"):
            output = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    output[i, j] = np.sum(img[i:i+3, j:j+3] * kernel)
            fig3 = px.imshow(output, color_continuous_scale="Viridis", text_auto=True, title="Output Map")
            fig3.update_coloraxes(showscale=False)
            st.plotly_chart(fig3, use_container_width=True)
            
    st.markdown("""
    ### 3.3 Advanced Architecture: Padding, Stride, ReLU, and Pooling
    Real CNNs do more than just convolve:
    * **Stride:** How many pixels the filter jumps when it slides. A stride of 2 halves the image size.
    * **Padding:** Adding fake 0-pixels around the edge of the image so the 3x3 filter doesn't fall off the edges, preserving spatial dimensions.
    * **ReLU (Rectified Linear Unit):** An activation function that simply changes all negative numbers in the feature map to zero. This introduces *non-linearity*, allowing the network to learn complex shapes.
    * **Pooling:** A downsampling layer (like Max Pooling) that takes the highest number in a 2x2 grid and throws the rest away. This saves memory and makes the network "spatially invariant" (it recognizes a dog ear whether it's shifted slightly left or right).
    
    When you mask an image in our lab, you are mathematically erasing the textures that the CNN's filters are desperately looking for!
    """)
    render_knowledge_check("ch3_q1", "What happens if you apply a Horizontal Edge filter to an image that only contains a Vertical Line?", 
        ["The output feature map shows a massive positive reaction.", "The output feature map will be mostly zeros because the filter didn't find its mathematical pattern.", "The filter automatically rotates to find the line."], 1, 
        "Filters are mathematically hardcoded matrices. If the pattern isn't there, the dot-product sum is near zero.")

def chapter_4():
    st.header("Chapter 4: How Image Classification Works")
    st.markdown("""
    ### 4.1 From Feature Maps to Logits
    After the image passes through dozens of convolutional and pooling layers, the final feature maps are flattened and passed to a **Classification Head** (standard dense layers). 
    
    This final layer outputs a raw list of numbers called **logits**. Our model outputs exactly 1,000 logits—one for every object category it knows.
    
    ### 4.2 The Softmax Function
    Logits are just unbounded numbers (e.g., `14.5` or `-2.1`). To make them readable, we pass them through the **Softmax** formula:
    
    $$\sigma(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^K e^{z_j}}$$
    
    Softmax does two vital things:
    1. It squashes all 1,000 numbers so they fall between `0.0` and `1.0`.
    2. It forces all 1,000 numbers to add up to exactly `1.0` (or 100%).
    
    These are our **Probability Scores**. We usually only care about the **Top-K** (e.g., Top 3 highest scores).
    
    ### 4.3 Confidence vs. Correctness
    If the model outputs 99% for 'Samoyed', it means the model's math is highly *confident*. **Confidence is not accuracy.** If you feed the model a picture of a spaceship (which it was never trained on), it might output 99% confidence for "Car" simply because it has wheels!
    """)
    render_knowledge_check("ch4_q1", "What does the Softmax function guarantee about the output scores?", 
        ["It guarantees the top prediction is factually correct.", "It guarantees all the output probabilities sum to exactly 100%.", "It removes all predictions below 50%."], 1, 
        "Softmax turns raw logits into a standardized probability distribution summing to 1.0.")

def chapter_5():
    st.header("Chapter 5: Meet MobileNetV2")
    st.markdown("""
    ### 5.1 Why MobileNetV2?
    Many deep learning models (like ResNet or VGG) are massive, requiring gigabytes of RAM and powerful GPUs to run. **MobileNetV2** was specifically designed by Google to run on mobile phones and lightweight CPUs (like the one running this application).
    
    ### 5.2 Depthwise Separable Convolutions
    Standard convolution is computationally expensive because it mixes spatial data (pixels) and channel data (RGB) all at once. MobileNetV2 uses a trick:
    1. **Depthwise Convolution:** It filters each color channel separately.
    2. **Pointwise Convolution:** It then uses a 1x1 filter to mix the channels together.
    This drastic reduction in multiplication operations is what makes the model so fast!
    
    ### 5.3 The ImageNet Dataset
    MobileNetV2 was trained on **ImageNet**, a famous dataset containing over 1.2 million images mapped to exactly 1,000 specific classes (like 'Samoyed', 'Keyboard', or 'Tench fish'). Our model can *only* predict these 1,000 things. If you upload a picture of a Pikachu, the model physically cannot output "Pikachu"—it will guess whatever ImageNet class looks most similar.
    """)
    render_knowledge_check("ch5_q1", "Why is MobileNetV2 able to run so fast on a standard CPU?", 
        ["It uses low-resolution black and white images only.", "It uses Depthwise Separable Convolutions to drastically reduce the number of mathematical operations.", "It only knows 10 object classes instead of 1000."], 1, 
        "By separating spatial filtering from channel mixing, MobileNetV2 achieves high accuracy with a fraction of the compute cost.")

def chapter_6():
    st.header("Chapter 6: How Masking Changes a Prediction")
    st.markdown("""
    ### 6.1 The Image Occlusion Experiment
    If a neural network relies on a specific visual feature (like a dog's snout) to make a prediction, what happens if we delete that feature?
    
    By drawing a grey box over a region of the image (**occlusion**), we mathematically overwrite that information. When we run the modified image through the model again, the Probability Score shifts.
    
    * If the score **drops heavily**, the masked region contained critical information.
    * If the score **increases**, the region actually contained distracting noise!
    
    ### 6.2 The Artifact Problem (Out of Distribution)
    There is a major catch to our experiment. When we draw a sharp grey square on an image, we introduce artificial, straight edges that weren't there before. 
    
    Because the model was never trained on images with floating grey boxes, this is an **Out-of-Distribution** input. We must be careful: did the score drop because we hid the dog, or because the model's filters got completely confused by the unnatural straight grey lines?
    """)
    render_knowledge_check("ch6_q1", "If you mask a region and the model's confidence goes UP, what does this imply?", 
        ["The model broke.", "The masked region likely contained distracting background noise, so removing it helped the model focus.", "The masked region contained the dog's face."], 1, 
        "Removing noisy backgrounds (like complex grass textures) can sometimes make the core pattern easier for the CNN to classify.")

def chapter_7():
    st.header("Chapter 7: Understanding the AI X-Ray (Heatmap)")
    st.markdown("""
    ### 7.1 The Occlusion Sensitivity Formula
    In the Lab, when you click "Generate Heatmap", the computer runs a specific mathematical formula to build the **AI X-Ray**.
    
    For a selected target class $c$ (like 'Samoyed'):
    1. $S_{original}(c) =$ The probability score of the unedited image.
    2. $S_{masked}(c, i) =$ The probability score when grid region $i$ is covered.
    
    The **Signed Sensitivity Delta (D)** for that region is calculated as:
    $$D(i) = S_{original}(c) - S_{masked}(c, i)$$
    
    ### 7.2 Interpreting the Math
    * You mask the dog's face. The score drops from 80% to 35%. 
      $D = 0.80 - 0.35 = 0.45$ (A large positive effect).
    * You mask the grass. The score rises from 80% to 82%. 
      $D = 0.80 - 0.82 = -0.02$ (A slight negative effect).
      
    ### 7.3 Absolute Effect
    Sometimes we only care about *magnitude*—how much did the region matter, regardless of whether it helped or hurt? We calculate the **Absolute Effect (A)**:          
    **A(i) = | D(i) |**
    
    We map these numbers to colors (Red for high drop, Blue for negative drop) to create the Heatmap matrix.
    """)
    render_knowledge_check("ch7_q1", "In our formula, what does a large positive D(i) value mean?", 
        ["Masking that region helped the model.", "Masking that region hurt the model's confidence significantly.", "The region is completely black."], 1, 
        "A positive D(i) means Original - Masked was a positive number, meaning the Masked score was much lower.")

def chapter_8():
    st.header("Chapter 8: Conducting Valid Experiments")
    st.markdown("""
    ### 8.1 The Scientific Method in the Lab
    When you go to the Interactive Lab, you are not just playing a game; you are conducting empirical experiments on a black-box AI.
    
    1. **Form a Hypothesis:** Before clicking, make a prediction. *"I hypothesize that masking the dog's ear will reduce the score more than masking its tail."*
    2. **Control the Variables:** Only change one grid cell at a time. If you mask the whole image at once, you can't isolate which feature caused the score to drop.
    3. **Observe & Record:** The Experiment Lab table at the bottom of the app will automatically log the independent variable (the cell you clicked) and the dependent variable (the new score).
    4. **Conclude Cautiously:** Look at the data before deciding if your hypothesis was right.
    """)
    render_knowledge_check("ch8_q1", "Why is it important to test one grid cell at a time when forming a hypothesis?", 
        ["Because the computer will crash if you click two.", "To isolate variables; changing multiple cells makes it impossible to know which specific region caused the score change.", "It's a rule to make the game harder."], 1, 
        "Controlling variables is fundamental to science. You must isolate the input to understand the output.")

def chapter_9():
    st.header("Chapter 9: Limitations, Bias & Responsible Interpretation")
    st.markdown("""
    ### 9.1 The Illusion of Understanding
    A heatmap is incredibly useful, but it has severe limitations. **It does not prove human reasoning.** 
    
    If the heatmap highlights a dog's face, it does not mean the CNN "understands" what a face is. It just means the raw pixel values in that area heavily activated the math in the deeper layers.
    
    ### 9.2 Dataset Bias (Spurious Correlations)
    Imagine a network was trained on thousands of images where cows are always standing on green grass. The network might learn that "Green Grass = Cow". 
    
    If you show it a cow on a beach, it might fail. In the lab, masking the grass might drastically lower the score, even though grass is not part of a cow! The model learned the *context*, not just the object.
    
    You must interpret heatmaps responsibly. They show you what the math relies on, not what the AI "thinks."
    """)
    render_knowledge_check("ch9_q1", "If the AI X-Ray is entirely red over a patch of snow behind a Husky, what does this prove?", 
        ["The model thinks snow is a dog.", "The model relies heavily on the background context (snow) to classify the image as a Husky.", "The heatmap calculation is broken."], 1, 
        "This is a classic dataset bias! If Huskies are usually photographed in snow, the network learns a spurious correlation: 'White Snow = High probability of Husky'.")

def chapter_10():
    st.header("Chapter 10: Putting It All Together")
    st.markdown("""
    ### The CNN Blind Spot Pipeline
    You are now ready for the lab. Here is exactly what happens under the hood when you use it:
    
    1. **Input:** You view the standardized image.
    2. **Preprocess:** The app resizes and normalizes the image tensor.
    3. **Baseline Inference:** MobileNetV2 calculates the original Softmax probabilities.
    4. **Interact:** You click a grid button to mask a region.
    5. **Re-evaluate:** The newly masked image goes through the network.
    6. **X-Ray Math:** The app calculates the sensitivity delta $D(i)$ for all regions by running inference 16 times in the background.
    7. **Visualize:** Plotly renders the results as a color-coded Heatmap.
    8. **Log:** Your actions and score changes are permanently recorded in the Experiment Lab CSV.
    """)
    st.success("You have finished the core curriculum! You may proceed to the Interactive Lab via the sidebar.")

def render_academy_page():
    st.title("📚 Learning Academy")
    st.markdown("A structured, interactive textbook to master Deep Learning concepts.")
    
    chapters = {
        "1. How a Computer Sees an Image": chapter_1,
        "2. What Is a Neural Network?": chapter_2,
        "3. Understanding CNNs": chapter_3,
        "4. Image Classification": chapter_4,
        "5. Meet MobileNetV2": chapter_5,
        "6. Masking & Predictions": chapter_6,
        "7. The AI X-Ray Math": chapter_7,
        "8. Conducting Valid Experiments": chapter_8,
        "9. Limitations & Bias": chapter_9,
        "10. The Full Pipeline": chapter_10
    }
    
    selection = st.sidebar.radio("Select Chapter:", list(chapters.keys()))
    st.divider()
    chapters[selection]()
