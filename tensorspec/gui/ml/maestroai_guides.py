from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QListWidget, QTextBrowser)
from PySide6.QtCore import Qt

class SSLGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Guide to SSL Models")
        self.resize(700, 450)
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.menu_list = QListWidget()
        self.menu_list.addItems([
            "1. What is SSL?", "2. Autoencoder (AE)", "3. Beta-VAE", 
            "4. MAE (Masked Autoencoder)", "5. ViT-MAE (Transformer)", 
            "6. SimCLR", "7. MoCo", "8. BYOL", "9. SwAV"
        ])
        
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        
        splitter.addWidget(self.menu_list)
        splitter.addWidget(self.info_display)
        splitter.setSizes([200, 500])
        layout.addWidget(splitter)
        
        self.explanations = {
            0: "<h3>What is Self-Supervised Learning (SSL)?</h3><p>In traditional machine learning, you have to manually label thousands of images. <b>SSL learns by itself.</b></p><p>Instead of relying on human labels, SSL algorithms play mathematical 'games' with your raw ARPES data to discover the underlying physics on their own.</p><p>Click on the algorithms on the left to see exactly how each 'game' works and when you should use them for your data.</p>",
            1: "<h3>Autoencoder (AE)</h3><p><b>The Game:</b> Data Compression.</p><p><b>How it works:</b> It squeezes your 2D ARPES map into a tiny mathematical bottleneck, and then tries to perfectly redraw the original image from that squeezed data.</p><p><b>Best for:</b> A great, fast baseline. It learns the most visually dominant features in your data (like the brightest bands or the Fermi edge).</p>",
            2: "<h3>Beta-VAE</h3><p><b>The Game:</b> Smooth & Independent Compression.</p><p><b>How it works:</b> Like an Autoencoder, but it forces the bottleneck to be mathematically 'smooth'. It tries to make every variable independent (e.g., Variable 1 controls binding energy shifts, Variable 2 controls intensity).</p><p><b>Best for:</b> When you want to smoothly track a specific band shifting across a real-space map without abrupt jumps.</p>",
            3: "<h3>MAE (Masked Autoencoder)</h3><p><b>The Game:</b> The Missing Puzzle Pieces.</p><p><b>How it works:</b> It blacks out 75% of your ARPES image and forces the network to guess the missing pixels based on the surviving 25%.</p><p><b>Best for:</b> Extracting deep physics. To guess a missing piece of a band, the AI *must* learn the actual physical dispersion rules, ignoring random background noise.</p>",
            4: "<h3>ViT-MAE (Vision Transformer)</h3><p><b>The Game:</b> The Missing Puzzle Pieces (Global Scale).</p><p><b>How it works:</b> The same puzzle game as the MAE, but it uses 'Attention' instead of standard convolutions. It looks at the *entire* image at once.</p><p><b>Best for:</b> Finding long-range correlations, like a band at -10° physically coupling to a band at +10° across the Brillouin zone.</p>",
            5: "<h3>SimCLR</h3><p><b>The Game:</b> Find the Clones.</p><p><b>How it works:</b> It takes a single ARPES cut, makes two slightly distorted copies of it, and tells the AI: <i>'These two are the same, pull them together! Everything else in this batch is different, push them away!'</i></p><p><b>Best for:</b> Highly diverse datasets where you want the AI to learn that slightly noisy or shifted bands are actually the same physical state.</p>",
            6: "<h3>MoCo (Momentum Contrast)</h3><p><b>The Game:</b> The Massive Memory Bank.</p><p><b>How it works:</b> It plays the same clone-matching game as SimCLR, but it keeps a 'memory bank' of thousands of previous ARPES images to compare against, rather than just the current small batch.</p><p><b>Best for:</b> Very noisy beamline data. The massive memory bank stabilizes the learning process immensely.</p>",
            7: "<h3>BYOL</h3><p><b>The Game:</b> Teacher & Student.</p><p><b>How it works:</b> A 'Student' network tries to predict exactly what a slower, more stable 'Teacher' network is looking at. It does *not* push different images away from each other.</p><p><b>Best for:</b> Datasets where all your ARPES cuts look extremely similar (e.g., a highly uniform sample). Because it doesn't push different images apart, it won't accidentally separate similar physics.</p>",
            8: "<h3>SwAV</h3><p><b>The Game:</b> The Sorting Hat.</p><p><b>How it works:</b> Instead of comparing images to other images, it sets up a number of idealized 'Prototypes' (buckets). It forces two distorted views of the same ARPES cut to fall into the exact same bucket.</p><p><b>Best for:</b> Direct Clustering! If your ultimate goal is to run K-Means to find distinct structural domains, SwAV is explicitly designed to group your data into distinct phases.</p>"
        }
        self.menu_list.currentRowChanged.connect(self.update_info)
        self.menu_list.setCurrentRow(0)

    def update_info(self, index):
        self.info_display.setHtml(self.explanations.get(index, "Select an item to see details."))


class ClusterGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Guide to Clustering")
        self.resize(700, 450)
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.menu_list = QListWidget()
        self.menu_list.addItems([
            "1. What is Clustering?", "2. K-Means", "3. Gaussian Mixture",
            "4. Hierarchical", "5. DBSCAN (What is eps?)", "6. What is UMAP?"
        ])
        
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        
        splitter.addWidget(self.menu_list)
        splitter.addWidget(self.info_display)
        splitter.setSizes([200, 500])
        layout.addWidget(splitter)
        
        self.explanations = {
            0: "<h3>What is Clustering?</h3><p>Once your Neural Network has extracted the physics features from the ARPES maps, <b>Clustering algorithms group the pixels together based on mathematical similarity.</b></p><p>This allows you to automatically extract spatial domains, charge density wave (CDW) regions, or distinct structural phases without manually drawing boxes on the map.</p>",
            1: "<h3>K-Means</h3><p><b>The Game:</b> Rigid, spherical buckets.</p><p><b>How it works:</b> You tell it exactly how many clusters you want (the <b>'k'</b> value). It drops <i>k</i> centers into the data and assigns every pixel to the nearest center. It assumes all domains are roughly the same size and spherical.</p><p><b>Best for:</b> Clean data where you know exactly how many physical phases are present on the sample.</p>",
            2: "<h3>Gaussian Mixture</h3><p><b>The Game:</b> Flexible, overlapping ellipses.</p><p><b>How it works:</b> Like K-Means, you define the number of clusters (<b>k</b>). However, it allows the clusters to stretch into ovals and overlap. It assigns probabilities rather than hard boundaries.</p><p><b>Best for:</b> Data where the boundaries between domains are fuzzy, or where one physical domain has much more variation than another.</p>",
            3: "<h3>Hierarchical</h3><p><b>The Game:</b> The Family Tree.</p><p><b>How it works:</b> It starts by treating every single pixel as its own cluster, and then slowly merges the most similar ones together step-by-step until it hits your target number of clusters (<b>k</b>).</p><p><b>Best for:</b> Discovering nested physics, like finding subtle sub-domains hiding inside a larger macroscopic phase.</p>",
            4: "<h3>DBSCAN (Density-Based Spatial Clustering)</h3><p><b>The Game:</b> The Contagious Blob.</p><p><b>How it works:</b> It does NOT need you to guess the number of clusters! Instead, it finds dense \"blobs\" of similar pixels. It separates the high-density areas from the low-density noise.</p><p><b>What is 'eps'?</b> Epsilon (eps) is the <b>Search Radius</b>. If eps is 1.5, the algorithm looks 1.5 units around a pixel. If it finds enough neighbors, it starts a cluster. <br><br>&bull; <b>Too High:</b> Everything merges into one giant cluster.<br>&bull; <b>Too Low:</b> The algorithm thinks everything is random noise, and no clusters form (everything turns black). Adjust this slider until the distinct phases 'snap' into view!</p>",
            5: "<h3>What is UMAP?</h3><p><b>The Game:</b> 2D Translation.</p><p><b>How it works:</b> Your neural networks output data in 64 or 128 dimensions. Humans cannot see in 64 dimensions! UMAP is a mathematical translator that perfectly squashes those 64 dimensions down to a 2D scatter plot, preserving the physical relationships so you can visually click on them.</p>"
        }
        self.menu_list.currentRowChanged.connect(self.update_info)
        self.menu_list.setCurrentRow(0)

    def update_info(self, index):
        self.info_display.setHtml(self.explanations.get(index, "Select an item to see details."))


class SupGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Guide to Supervised Learning")
        self.resize(700, 450)
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.menu_list = QListWidget()
        self.menu_list.addItems([
            "1. What is Supervised Learning?", "2. Few-Shot Learning", 
            "3. Step 1: Define Labels", "4. Step 2: Collect Data", 
            "5. Step 3: Train & Test", "6. Reading the Results"
        ])
        
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        
        splitter.addWidget(self.menu_list)
        splitter.addWidget(self.info_display)
        splitter.setSizes([200, 500])
        layout.addWidget(splitter)
        
        self.explanations = {
            0: "<h3>What is Supervised Learning?</h3><p>Unlike Self-Supervised Learning (SSL) which groups data automatically, <b>Supervised Learning requires a human teacher.</b></p><p>You manually identify a few specific physics features (like a superconducting gap or a Dirac cone), and the AI learns exactly what those specific features look like so it can find them everywhere else on the sample.</p>",
            1: "<h3>Few-Shot Learning</h3><p><b>The Problem:</b> Standard neural networks need thousands of labeled examples to learn. You don't have time to click 10,000 pixels!</p><p><b>The Solution:</b> MAESTRO uses a customized, lightweight <i>Few-Shot CNN</i>. Because it is highly optimized for ARPES data, it only needs you to click about <b>5 to 15 examples</b> per class to successfully learn the pattern.</p>",
            2: "<h3>Step 1: Define Target Labels</h3><p>Decide how many distinct physical phases you want to extract.</p><p>&bull; <b>Example (2 Labels):</b> Label 1 = \"Gapped Phase\", Label 2 = \"Metallic Phase\".<br>&bull; <b>Example (3 Labels):</b> Label 1 = \"Domain A\", Label 2 = \"Domain B\", Label 3 = \"Junk/Background\".</p>",
            3: "<h3>Step 2: Collect Data (The Crosshairs)</h3><p>This is where you teach the AI.</p><p>1. Look at the <b>Interactive 4D Viewer</b>.<br>2. Move the <b>X and Y Center</b> sliders to physically move the crosshair to a known domain on the sample.<br>3. Click the <b>Assign Target Coordinate</b> button for the correct label.<br>4. Repeat until you have 5-15 counts for each label.</p>",
            4: "<h3>Step 3: Train & Test</h3><p><b>Train Model:</b> The AI studies the specific energy-momentum dispersions at the X/Y coordinates you just provided.</p><p><b>Test (Classify Entire Map):</b> The AI takes what it learned and applies it to <i>every single pixel</i> in the dataset, automatically mapping out the spatial distribution of your target phases.</p>",
            5: "<h3>Reading the Results</h3><p>Once inference is complete, the Spatial Map will switch to a <b>Probability Map</b>.</p><p>Instead of hard boundaries, the AI uses RGB color mixing to show its confidence. If a pixel is 80% Label 1 (Blue) and 20% Label 2 (Red), it will appear purple. You can move the crosshair over any pixel to read the exact percentage breakdown in the plot title!</p>"
        }
        
        self.menu_list.currentRowChanged.connect(self.update_info)
        self.menu_list.setCurrentRow(0)

    def update_info(self, index):
        self.info_display.setHtml(self.explanations.get(index, "Select an item to see details."))


class MasterGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MAESTRO Workflow Guide")
        self.resize(750, 500)
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.menu_list = QListWidget()
        self.menu_list.addItems([
            "Workflow Overview", "Step 1: Load Data", "Step 2: Interactive Viewer", 
            "Step 3: Choose Your ML Path", "Path A: Self-Supervised & Clustering", 
            "Path B: Supervised (Few-Shot)", "Step 4: View & Export Results"
        ])
        
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        
        splitter.addWidget(self.menu_list)
        splitter.addWidget(self.info_display)
        splitter.setSizes([220, 530])
        layout.addWidget(splitter)
        
        self.explanations = {
            0: "<h3>Welcome to the MAESTRO Suite!</h3><p>This app is designed to help you extract physical domains, charge density waves, and structural phases from massive 4D ARPES datasets (X, Y, Energy, Angle).</p><p><b>General Workflow:</b><br>1. Load a file from your disk.<br>2. Inspect the raw data in the 4D Viewer.<br>3. Choose a Machine Learning path (Supervised vs. Unsupervised).<br>4. Export the resulting map back to a CSV for plotting.</p><p><i>Click through the steps on the left to learn how to navigate the app!</i></p>",
            1: "<h3>Step 1: Load Data</h3><p>Look at the left panel.</p><p><b>1. Select Folder:</b> Choose the directory containing your `.h5` Scienta ARPES scans.<br><b>2. Choose File:</b> Click a specific file from the list.<br><b>3. Load to Workspace:</b> This pushes the massive file into your Mac's RAM. Once it appears in the <b>RAM Workspace</b> list, click it to activate it and view it!</p>",
            2: "<h3>Step 2: The Interactive 4D Viewer</h3><p>This is your main command center.</p><p>&bull; <b>Spatial Map (Left Plot):</b> Shows your sample in real space (X vs. Y).<br>&bull; <b>Dispersion Map (Right Plot):</b> Shows the ARPES physics (Energy vs. Angle) at the exact pixel you are hovering over.</p><p><b>Sliders:</b> Use the X/Y sliders to move the crosshair around the spatial map. Use the Energy/Angle sliders to isolate a specific band or Fermi surface.</p>",
            3: "<h3>Step 3: Choose Your ML Path</h3><p>How do you want to extract your spatial domains?</p><p><b>Path A: The Unknown (SSL & Clustering).</b> Use this if you don't know what domains exist on your sample and want the AI to discover them for you. (Uses the first two tabs on the right).</p><p><b>Path B: The Known (Supervised Learning).</b> Use this if you already know exactly what phases exist (e.g., you can visually see a metallic phase and an insulating phase) and just want to map where they are. (Uses the 3rd tab on the right).</p>",
            4: "<h3>Path A: Self-Supervised & Clustering</h3><p><b>1. Embed the Data (SSL Tab):</b> Run a neural network (like SimCLR or MAE) to mathematically compress your ARPES bands into 'features'.<br><i>(Note: You can skip this and just use raw EDC/MDC integration instead!)</i></p><p><b>2. Group the Pixels (Clustering Tab):</b> Select your Embedding (or Viewer EDC/MDC), pick an algorithm like K-Means or DBSCAN, and click Run. The app will automatically group similar pixels into color-coded domains!</p>",
            5: "<h3>Path B: Supervised (Few-Shot) Learning</h3><p><b>1. Define Labels:</b> Tell the app how many distinct phases you are looking for.<br><b>2. Teach the AI:</b> Use the X/Y sliders in the 4D Viewer to target a specific phase, then click the 'Assign' button to record that pixel. Give the AI 5-10 examples per phase.<br><b>3. Train & Test:</b> The AI learns the physics of your targets and classifies the entire map, generating a Probability Map showing its confidence.</p>",
            6: "<h3>Step 4: View & Export Results</h3><p><b>Viewing:</b> Once an ML pipeline finishes, look above the Spatial Map. Change the <b>Spatial Map Display</b> dropdown from 'Intensity' to your new Clustering Domains or Supervised Probabilities to see the physical map!</p><p><b>Exporting:</b><br>&bull; In the Clustering tab, click <b>Save Labels to CSV/TXT</b>.<br>&bull; In the Supervised tab, click <b>Save Classification Results</b>.<br>These buttons output clean, flattened arrays that you can immediately drag into OriginLab, Igor Pro, or Python scripts for publication plotting.</p>"
        }
        
        self.menu_list.currentRowChanged.connect(self.update_info)
        self.menu_list.setCurrentRow(0)

    def update_info(self, index):
        self.info_display.setHtml(self.explanations.get(index, "Select an item to see details."))


class ActiveLearningGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Guide to Active Learning")
        self.resize(750, 500)
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.menu_list = QListWidget()
        self.menu_list.addItems([
            "1. What is Active Learning?", "2. How to Read the Maps",
            "3. Bayesian Network (GPU)", "4. Deep Ensembles (GPU)",
            "5. Evidential DL (GPU)", "6. Gaussian Process (CPU)",
            "7. Random Forest (CPU)"
        ])
        
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        
        splitter.addWidget(self.menu_list)
        splitter.addWidget(self.info_display)
        splitter.setSizes([220, 530])
        layout.addWidget(splitter)
        
        self.explanations = {
            0: "<h3>What is Active Learning?</h3><p><b>The Goal: Stop wasting beamline time!</b></p><p>Instead of exhaustively rastering a massive 10,000-point grid, you can take a fast, low-resolution 'scout' scan. Active Learning uses AI to predict what the rest of the sample looks like, and more importantly, it calculates exactly where it is <b>most confused</b>.</p><p>By steering the motors directly to the regions of highest mathematical uncertainty, you map phase boundaries perfectly using a fraction of the measurements.</p>",
            1: "<h3>How to Read the Maps</h3><p>When you run an algorithm, you get two plots:</p><p>&bull; <b>Phase Prediction:</b> The AI's best guess of what the spatial domain map looks like, expanding 20% beyond your actual scanned area.</p><p>&bull; <b>Uncertainty Heatmap:</b> This is your steering map! Bright regions represent high doubt. <br> - Bright spots <i>inside</i> the dashed box mean the AI is unsure about a phase boundary. Do a high-res sub-scan there!<br> - Bright spots <i>outside</i> the dashed box indicate a physical phase bleeding off the edge. Move the manipulator in that direction!</p>",
            2: "<h3>Bayesian Network (GPU)</h3><p><b>The Game:</b> Quantum Multiverse.</p><p><b>How it works:</b> It trains a neural network on your sample, but leaves 'Dropout' turned on during prediction. It predicts the map 50 separate times, randomly turning off different neurons each time. Uncertainty is measured by how much the 50 'parallel universes' disagree with each other.</p><p><b>Best for:</b> Lightning-fast uncertainty mapping using Apple Silicon.</p>",
            3: "<h3>Deep Ensembles (GPU)</h3><p><b>The Game:</b> The Board of Directors.</p><p><b>How it works:</b> It literally trains 5 completely independent neural networks from scratch. They vote on what the phase map should look like. Uncertainty is measured by the variance between their votes.</p><p><b>Best for:</b> The current industry gold-standard for AI uncertainty. Very robust, but takes 5x longer to train.</p>",
            4: "<h3>Evidential Deep Learning (GPU)</h3><p><b>The Game:</b> Explicit Doubt.</p><p><b>How it works:</b> Instead of predicting a probability, the network predicts 'Evidence' (a Dirichlet distribution). If it has never seen a specific type of spectrum before, it outputs zero evidence for all classes, explicitly flagging it as unknown.</p><p><b>Best for:</b> Discovering completely new physics. It is the only algorithm that knows when it is looking at an 'Out-of-Distribution' phase.</p>",
            5: "<h3>Gaussian Process (CPU)</h3><p><b>The Game:</b> The Mathematical Classic.</p><p><b>How it works:</b> Uses pure probability and matrix inversions to interpolate the spaces between your measured points. It is completely deterministic.</p><p><b>Best for:</b> Small datasets. (Note: GP math scales cubically, so it will randomly subsample your data to ~1,200 points to prevent crashing your Mac's RAM).</p>",
            6: "<h3>Random Forest (CPU)</h3><p><b>The Game:</b> The Wisdom of Crowds.</p><p><b>How it works:</b> Builds 100 simple 'Decision Trees', each looking at a slightly different random subset of your data. Uncertainty is the disagreement among the 100 trees.</p><p><b>Best for:</b> Very noisy data where you want a fast, traditional algorithm without relying on deep learning.</p>"
        }
        
        self.menu_list.currentRowChanged.connect(self.update_info)
        self.menu_list.setCurrentRow(0)

    def update_info(self, index):
        self.info_display.setHtml(self.explanations.get(index, "Select an item to see details."))


class SimulateALGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Guide to Simulate AL")
        self.resize(650, 400)
        
        layout = QVBoxLayout(self)
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        layout.addWidget(self.info_display)
        
        intro_html = """
        <h3>What is the 'Simulate AL' Tab?</h3>
        <p><b>The Goal: Prove that the AI actually saves time!</b></p>
        <p>Before you trust an AI to drive physical motors on a real beamline, you need to know it works. This tab lets you load a massive, fully completed spatial map from your hard drive, but <i>hides</i> it from the AI.</p>
        <p>It gives the AI a tiny handful of random pixels as a "scout scan," and forces the AI to iteratively ask for the points it is most confused about. As it asks, the tab reveals the true pixel.</p>
        
        <h3>How to use it:</h3>
        <ul>
            <li><b>1. Initialize Simulation:</b> Grabs a small randomized starting grid from your data.</li>
            <li><b>2. Simulate 1 Step:</b> The AI calculates uncertainty, picks the single best coordinate to measure next, and reveals it.</li>
            <li><b>3. Fast-Forward:</b> Puts the loop on autopilot so you can watch the AI intelligently reconstruct the boundaries of your domains in real-time, saving up to 80% of the pixels!</li>
        </ul>
        """
        self.info_display.setHtml(intro_html)


class AlignmentGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Guide to 3D Alignment")
        self.resize(750, 550)
        
        layout = QVBoxLayout(self)
        self.info_display = QTextBrowser()
        self.info_display.setOpenExternalLinks(True)
        layout.addWidget(self.info_display)
        
        intro_html = """
        <h3>What is the '3D Alignment' Tab?</h3>
        <p><b>The Goal: Map the physical deformations of your crystal domains.</b></p>
        <p>This tool assumes the intrinsic Fermi surface is the same everywhere on your sample, but different microscopic domains might be physically deformed. You can run two different engines depending on the physics of your sample:</p>
        
        <hr>
        
        <h3>Mode 1: Azimuthal Twist (In-Plane)</h3>
        <p><b>Use case:</b> Twinned domains, randomly oriented flakes, or twisted bilayers.</p>
        <ul>
            <li><b>The Math:</b> The app extracts a 1D dispersion cut passing straight through the Gamma point of the reference map. It then rotates this virtual cut a full 360 degrees.</li>
            <li><b>The Output:</b> It finds the exact in-plane rotation angle (&phi;) that perfectly fits your local (X, Y) data.</li>
        </ul>
        
        <h3>Mode 2: Surface Normal Tilt (Out-of-Plane)</h3>
        <p><b>Use case:</b> Bowed/wrinkled samples, or samples mounted on uneven surfaces.</p>
        <ul>
            <li><b>The Math:</b> The app assumes the crystal is <i>not</i> twisted, but instead tilts away from the detector. It searches through every Deflection slice and horizontal Slit shift in the 3D reference volume to find the perfect correlation match.</li>
            <li><b>The Output:</b> Two maps showing exactly how far the local surface normal has tilted away from the reference Gamma point (measured in pixels/degrees along both the Slit and Deflection axes).</li>
        </ul>
        
        <hr>
        
        <h3>How to use it:</h3>
        <ol>
            <li><b>Define the Gamma Point:</b> Use the 4D Viewer to find the exact coordinates of the Gamma point on your high-statistics reference Fermi map. Enter those coordinates in the Gamma Target boxes.</li>
            <li><b>Select Mode:</b> Choose Azimuthal Twist or Normal Tilt from the dropdown.</li>
            <li><b>Run:</b> Click 'Run Global Alignment Search' and let the matrix math do the rest!</li>
        </ol>
        """
        self.info_display.setHtml(intro_html)