## Document Extraction with Object Detection

An Object detection system to detect and extract documents from the given Image. The model used is the 'SSD-MOBILENET-V1',which has processing speed of 48ms and a Mean Average Precision (mAP) of 29.

<div float="left" align="center">
<img src="/assets/example.png"  width="80%"/>
</div>


## 1] Run Inference

To run an Inference over a single Image, open the inference.py file and set the file path, then execute the script.

1. Git clone the repository on your system. This will download the pre-trained model and required files on your computer.
```
git clone https://github.com/deepeshdm/Document-Extraction.git
```

2. Install the required dependencies to run the app
```
pip install -r requirements.txt
```

3. Open inference.py file and set the parameters and execute the file.

```
python inference.py
```



## 2] Web Interface

The Web Interface is made using streamlit, you can run it locally by executing the following commands :

**🔥 Official Website :** https://deepeshdm-document-extraction-web-interface-bz0qd3.streamlitapp.com/

<div align="center">
  <img src="/assets/web.png"  width="80%"/>
</div>
<br/>


1. Git clone the repository on your system. This will download the pre-trained model and required files on your computer.
```
git clone https://github.com/deepeshdm/Document-Extraction.git
```

2. Install the required dependencies to run the app
```
pip install -r requirements.txt
```

3. Start the streamlit server on specified port

```
streamlit run web_interface.py --server.port 4000
```


