# Reproducing Example Usage Scenarios of the BioImage.IO Chatbot Figure 2

This section provides detailed instructions for reproducing the example usage scenarios of the BioImage.IO Chatbot illustrated in Figure 2 of the main text:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vTIRwRldQBnTFqz0hvS01znGOEdoeDMJmZC-PlBM-O59u_xo7DfJlUEE9SlRsy6xO1hT2HuSOBrLmUz/pub?w=1324&amp;h=1063">


These steps will guide users through querying documents, utilizing online services, executing AI models, and developing extensions.

## Access the BioImage.IO Chatbot Interface
Launch the chatbot through the BioImage.IO website [here](https://bioimage.io/chat/) or use the dedicated user interface.

### Scenario (a): Querying Bioimage Analysis Documentation

- **Initiate a Query**: Type a question related to bioimage analysis, e.g., "What are the best practices for optimizing model performance on bioimage.io?"
- **Review the Chatbot's Response**: The chatbot will provide an answer that includes information extracted from the BioImage Model Zoo documentation.

### Scenario (b): Querying Online Services

- **Pose a Research-Related Query**: Ask the chatbot to find cell images at the G1 phase by typing "Please, find cell Images at G1 phase."
- **Interpret the Results**: The chatbot will respond with relevant studies, such as the one titled "DeepCycle: Deep learning reconstruction of the closed cell cycle trajectory from single-cell unsegmented microscopy images."

### Scenario (c): Running AI Models for Image Analysis

- **Download Image Data**: Begin by creating a new folder on your computer named test-image. Download the image data file from [this link](https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/bioengine-support/docs/screenshots/nuclei.tiff) and save it into the test-image folder.
- **Initiate Bioimage Analyst**: Navigate to the BioImage.IO chatbot interface at [this page](https://bioimage.io/chat/). Activate the Bioimage Analyst chatbot, known as "Bioimage Analyst(Bridget)", by selecting it from the chatbot menu on the right side of the page.
- **Mount your Data Folder**: Within the chat interface, click on the "Mount Files" button located below the dialog window. This action will allow you to mount the test-image folder that contains your downloaded image data. The chatbot will confirm the successful mounting of the folder and will list the files contained within, ensuring that your data is ready for analysis.
- **Run CellPose Model**: Type "Segment the image `/mnt/nuclei.tif` using Cellpose" to run the Cellpose model on the image data. Upon successful execution of the model, the chatbot will notify you that the segmentation process is complete and will display the analyzed results.

### Scenario (d): Developing New Extensions

Follow the steps below to develop a new extension for microscope stage control and image capture. For a detailed tutorial, visit our [GitHub repository](https://github.com/bioimage-io/bioimageio-chatbot/blob/main/docs/bioimage-chatbot-extension-tutorial.ipynb) or access the Jupyter Notebook directly through ImJoy [here](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1).

#### Step-by-Step Guide:

1. **Setup the Developer Environment**: Open a Jupyter Notebook. Install and import the `imjoy_rpc` and `pydantic` packages.
2. **Define Input Schemas**: Create classes for `MoveStageInput` and `SnapImageInput` to structure the user input.
3. **Implement Control Functions**: Write asynchronous functions `move_stage` and `snap_image`.
4. **Setup Extension Interface**: Develop the extension interface and define a schema getter function.
5. **Register the Extension**: Use `registerExtension` to add the new extension to the chatbot.

#### Example Query for Testing the Extension:

- Instruct the chatbot: "Move the stage by 5 mm to the right and take an image with an exposure of 15 milliseconds."
- The chatbot executes `move_stage` with `{ "x": 5, "y": 0 }` and `snap_image` with `{ "exposure": 15 }`.
- The chatbot confirms: "The stage was successfully moved 5 mm to the right, and an image was taken with an exposure time of 15 milliseconds."
