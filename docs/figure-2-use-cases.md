# Reproducing Example Usage Scenarios of the BioImage.IO Chatbot Figure 2

This section provides detailed instructions for reproducing the example usage scenarios of the BioImage.IO Chatbot illustrated in Figure 2 of the main text:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vTIRwRldQBnTFqz0hvS01znGOEdoeDMJmZC-PlBM-O59u_xo7DfJlUEE9SlRsy6xO1hT2HuSOBrLmUz/pub?w=1324&amp;h=1063">


These steps will guide users through querying documents, utilizing online services, executing AI models, and developing extensions.

## Access the BioImage.IO Chatbot Interface
Launch the chatbot through the BioImage.IO website [here](https://bioimage.io/chat/) or use the dedicated user interface.

## Video for Reproducing the Scenarios
 * **[A video showcasing information retrieval (as described in senario a-c)](https://zenodo.org/records/10967840/files/Supplementary-Video-1-bioimageio-chatbot-information-retrieval.mp4?download=1)**
 * **[A video showcasing AI model execution (as described in senario d)](https://zenodo.org/records/10967840/files/Supplementary-Video-2-bioimageio-chatbot-ai-image-analysis.mp4?download=1)**


### Scenario (a): Querying Bioimage Analysis Documentation

- **Initiate a Query**: Type a question related to bioimage analysis, e.g., "What are the best practices for optimizing model performance on bioimage.io?"
- **Review the Chatbot's Response**: The chatbot will provide an answer that includes information extracted from the BioImage Model Zoo documentation.

### Scenario (b): Exploring the Human Protein Atlas

- **Initiate a Query**: Ask the chatbot to find protein information in the Human Protein Atlas by typing "Tell me about PML protein and show me the cell images"
- **Interpret the Results**: The chatbot will respond by constructing an API call to the Protein Atlas database and displaying the relevant information about the PML protein, including cell images.

### Scenario (c): Querying the BioImage Archive

- **Initiate a Query**: Ask the chatbot to find cell images at the G1 phase by typing "Please, find datasets of cell images at G1 phase."
- **Interpret the Results**: The chatbot will initiate an API call to the BioImage Archive server, and return results such as a study titled "DeepCycle: Deep learning reconstruction of the closed cell cycle trajectory from single-cell unsegmented microscopy images."

### Scenario (d): Running AI Models for Image Analysis

- **Prereqsitues**: Ensure you have Chrome or a Chromium-based browser installed on your computer.
- **Download Image Data**: Begin by creating a new folder on your computer named `test-images`. Download the image data file from [this link](https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/example-data/nuclei.tif) and save it into the `test-images` folder.
- **Initiate Bioimage Analyst**: Navigate to the BioImage.IO chatbot interface at https://bioimage.io/chat/. Note that only Chrome or Chromium-based browser is supported at the moment. Select "Bioimage Analyst(Bridget)" located in the upper right corner of the chatbot interface.
- **Mount your Data Folder**: Within the chat interface, click on the "Mount Files" button located below the dialog window. This action will allow you to mount the test-image folder that contains your downloaded image data. The chatbot will confirm the successful mounting of the folder, you can now ask it to list the files contained within, and ensuring that your data is ready for analysis.
- **Perform segmentation using Cellpose model**: Type "Segment the image `/mnt/nuclei.tif` using Cellpose" to run the Cellpose model on the image data. Upon successful execution of the model, the chatbot will notify you that the segmentation process is complete and will display the analyzed results. Optionally, you can ask it to "count the number of nuclei in the image" if successfully segmented, "plot the size distribution of nuclei", or you can tell it to "use the visual inspection tool to analyze the figure and create a report about the size distribution".

### Scenario (e): Developing New Extensions

Follow the steps below to develop a new extension for microscope stage control and image capture. For a detailed tutorial, visit our [GitHub repository](https://github.com/bioimage-io/bioimageio-chatbot/blob/main/docs/bioimage-chatbot-extension-tutorial.ipynb) or access the Jupyter Notebook directly through ImJoy [here](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1).

### Scenario (f): Controlling a Microscope Stage and Capturing Images

- **Pre-requisites**: You will need a microscope and the squid control software

- **Create microscope extension**: Following the example in the above [chatbot extension example notebook](https://imjoy-notebook.netlify.app/lab/index.html?load=https://raw.githubusercontent.com/bioimage-io/bioimageio-chatbot/main/docs/bioimage-chatbot-extension-tutorial.ipynb&open=1), create a hypha service extension for controlling the microscope:
    1. **Setup the Developer Environment**: Open a Jupyter Notebook. Install and import the `imjoy_rpc`, `hypha_rpc` and `pydantic` packages.
    2. **Define Input Schemas**: Create classes for `MoveStageInput` and `SnapImageInput` to structure the user input. (Note: To help the chatbot understand the "center", you will need to tell the chatbot about the boundaries of the stage via the docstring of the `MoveStageInput` class)
    3. **Implement Control Functions**: Write asynchronous functions `move_stage` and `snap_image`.
    4. **Setup Extension Interface**: Develop the extension interface and define a schema getter function.
    5. **Register the Extension**: Register the extension as hypha server and connect to the the chatbot.
- **Initiate a Query**: Ask the chatbot to "Please move to the center and snap an image".
- **Interpret the Results**: The chatbot will execute the `move_stage` function to move the microscope stage to the center and then capture an image using the `snap_image` function. The chatbot will confirm the successful completion of the tasks.
