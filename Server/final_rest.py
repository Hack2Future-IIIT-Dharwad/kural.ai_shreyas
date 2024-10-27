from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf
from pydub import AudioSegment
import os
import requests
import base64
from groq import Groq
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and processors globally
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("shReYas0363/whisper-tiny-fine-tuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
groq_client = Groq(api_key='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')

def initialize_rag(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create Groq LLM
    llm = ChatGroq(
        groq_api_key='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',
        model_name="llama-3.2-1b-preview"
    )
    
    # Create prompt template
    template = """You are are medical assistant who's specialized function is to assist pregnant women during their period of pregnancy. You will be assisting pregnant women with their queries and you need to be accurate with the results. If not sure, ask questions for more guidance or play it safely by recommending the user to check with a professional doctor. You should be very polite and must reply with empathy. The reply should be like a human conversation and natural sounding. Keep the reply simple and straight-forward. Keep the reply very short within 1 sentences maximum no matter what. The reply should be of 1 short sentence only.
    
    Context: {context}
    
    Question: {question}
    """
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
   
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

PDF_PATH = "medical.pdf"  
qa_chain = initialize_rag(PDF_PATH)

def convertaudio_64(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return True

async def transcribe_audio(audio_path: str):
    try:
        audio_input, sampling_rate = sf.read(audio_path)
        
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        
        input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(transcription)
        return transcription
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

# #async def generate_chat_response(transcription: str):
#  #   try:
#   #      chat_completion = groq_client.chat.completions.create(
#    #         messages=[{"role": "user", "content": f'''You are a medical assistant give a very friendly one line response for this query: {transcription} Remember to give a single reply in just one reply'''}],
#             model="llama-3.2-1b-preview",
#         )
#         return chat_completion.choices[0].message.content
#     except Exception as e:
#         print(f"Error in chat response: {e}")
#         return None

# @app.post("/process-audio/")
# async def process_audio(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded file temporarily
#         input_path = file.filename
#         output_path = input_path.replace(".m4a", ".wav")
        
#         with open(input_path, "wb") as f:
#             f.write(await file.read())
        
#         # Convert audio and transcribe
#         convertaudio_64(input_path, output_path)
#         transcription = await transcribe_audio(output_path)
        
#         if transcription:
#             response = await generate_chat_response(transcription)
#             if response:
#                 # Get TTS audio as a response
#                 tts_url = "http://[::1]:5002/api/tts"
#                 tts_params = {"text": response, "speaker_id": "p374"}
#                 tts_response = requests.get(tts_url, params=tts_params)
                
#                 if tts_response.status_code == 200:
#                     output_audio_path = "response_audio.wav"
#                     with open(output_audio_path, "wb") as f:
#                         f.write(tts_response.content)
#                     with open(output_audio_path, "rb") as audio_file:
#                         encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
#                     return {"message": "Success", "chat": response, "encode": encoded_string, "transcription":transcription}
#                 else:
#                     raise HTTPException(status_code=500, detail="TTS service error.")
#             else:
#                 raise HTTPException(status_code=500, detail="Failed to generate chat response.")
#         else:
#             raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# @app.post("/process-audio/")
# async def process_audio(file: UploadFile = File(...), session_id: str = None):
#     try:
#         # Create or get session if session_id is provided
#         if not session_id:
#             session_id = base64.b64encode(os.urandom(16)).decode('utf-8')
        
#         session = chat_manager.get_or_create_session(session_id)
        
#         # Save the uploaded file temporarily
#         input_path = file.filename
#         output_path = input_path.replace(".m4a", ".wav")
        
#         with open(input_path, "wb") as f:
#             f.write(await file.read())
        
#         # Convert audio and transcribe
#         convertaudio_64(input_path, output_path)
#         transcription = await transcribe_audio(output_path)
        
#         if transcription:
#             # Get response using the session's conversation history
#             response = await session.get_response(transcription)
#             print("response it is: ",response)
            
#             if response:
#                 print("here is the problem")
#                 # Get TTS audio as a response
#                 tts_url = "http://[::1]:5002/api/tts"
#                 tts_params = {"text": response, "speaker_id": "p374"}
#                 tts_response = requests.get(tts_url, params=tts_params)
#                 print(tts_response)
#                 if tts_response.status_code == 200:
#                     print("TTS works")
#                     output_audio_path = "response_audio.wav"
#                     with open(output_audio_path, "wb") as f:
#                         f.write(tts_response.content)
#                     with open(output_audio_path, "rb") as audio_file:
#                         encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
#                     return {
#                         "message": "Success",
#                         "chat": response,
#                         "encode": encoded_string,
#                         "transcription": transcription,
#                         "session_id": session_id
#                     }
#                 else:
#                     raise HTTPException(status_code=500, detail="TTS service error.")
#             else:
#                 raise HTTPException(status_code=500, detail="Failed to generate chat response.")
#         else:
#             raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
async def transcribe_audio(audio_path: str):
    try:
        audio_input, sampling_rate = sf.read(audio_path)
        
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        
        input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(transcription)
        return transcription
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

async def generate_chat_response(transcription: str):
    try:
        # Use RAG chain instead of direct Groq API
        response = qa_chain({"query": transcription})
        return response['result']
    except Exception as e:
        print(f"Error in chat response: {e}")
        return None

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        input_path = file.filename
        output_path = input_path.replace(".m4a", ".wav")
        
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Convert audio and transcribe
        convertaudio_64(input_path, output_path)
        transcription = await transcribe_audio(output_path)
        
        if transcription:
            response = await generate_chat_response(transcription)
            if response:
                # Get TTS audio as a response
                tts_url = "http://[::1]:5002/api/tts"
                tts_params = {"text": response, "speaker_id": "p374"}
                tts_response = requests.get(tts_url, params=tts_params)
                
                if tts_response.status_code == 200:
                    output_audio_path = "response_audio.wav"
                    with open(output_audio_path, "wb") as f:
                        f.write(tts_response.content)
                    with open(output_audio_path, "rb") as audio_file:
                        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
                    return {"message": "Success", "chat": response, "encode": encoded_string, "transcription":transcription}
                else:
                    raise HTTPException(status_code=500, detail="TTS service error.")
            else:
                raise HTTPException(status_code=500, detail="Failed to generate chat response.")
        else:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Cleanup temporary files
        for file_path in [input_path, output_path, "response_audio.wav"]:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
