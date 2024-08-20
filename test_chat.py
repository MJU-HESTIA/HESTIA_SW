from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from PIL import Image, ImageTk
from openai import OpenAI
import google.generativeai as genai
import cv2
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from psycopg2 import sql
import psycopg2
import numpy as np
from groq import Groq
import threading
from datetime import datetime, timedelta
import time
import re
from io import BytesIO
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import pyaudio
import serial
import ollama

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 불러오기
groq_api_key = os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
weather_api_key = os.getenv("WEATHER_API_KEY")
db_name = os.getenv('DATABASE_NAME')
db_user = os.getenv('DATABASE_USER')
db_password = os.getenv('DATABASE_PASSWORD')
db_host = os.getenv('DATABASE_HOST')
db_port = os.getenv('DATABASE_PORT')

# PostgreSQL 연결 설정
try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        options="-c client_encoding=UTF8"
    )
    cursor = conn.cursor()
    print("데이터베이스 연결 성공")
except Exception as e:
    print(f"데이터베이스 연결 실패: {e}")

# SentenceTransformer 모델 로드
embedding_model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")

ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=1)

temp = 0
gas = 0 # gas 감지하면 1
light = 1 # 빛을 받으면 1 
previous_light = light
previous_gas = gas
temp_warning_given = False

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def read_from_esp32():
    global temp, gas, light
    
    while True:
        if ser.in_waiting > 0:
            try:
                # ESP32로부터 데이터 읽기
                data = ser.readline().decode('utf-8').rstrip()
            except UnicodeDecodeError:
                data = ser.readline().decode('utf-8', errors='ignore').rstrip()
            
            # 수신된 데이터가 올바른 형식인지 확인
            if data.startswith('<') and data.endswith('>'):
                data_type = data[1]
                data_value = data[3:-1]
                
                if data_type == 'T':  # 온도 데이터
                    try:
                        temp = float(data_value)
                        print(f"Temperature: {temp}°C")
                    except ValueError:
                        print(f"Invalid temperature data: {data_value}")
                
                elif data_type == 'G':  # 가스 데이터
                    gas = bool(int(data_value))
                    gas_status = "Gas detected" if gas else "No gas detected"
                    print(f"Gas Sensor: {gas_status}")
                
                elif data_type == 'L':  # 조도 데이터
                    light = bool(int(data_value))
                    light_status = "Light detected" if light else "No light detected"
                    print(f"Light Sensor: {light_status}")
                
                else:
                    print(f"Unknown data type: {data}")
            else:
                print(f"Invalid packet format: {data}")

# 조도 센서의 값을 지속적으로 모니터링하는 함수
def monitor_light():
    global light
    global previous_light
    global convo

    while True:
        # 현재 light 값을 계속 모니터링
        if previous_light == 1 and light == 0:
            response = "불을 끄셨네요 주무시나요?"
            convo.append({'role': 'assistant', 'content': response})
            print(f'ASSISTANT: {response}')
            speak_async(response)
        
        # 현재 값을 이전 값으로 업데이트
        previous_light = light
        
        # 잠시 대기 (이 시간을 조정하여 센서 데이터를 읽는 주기를 설정할 수 있음)
        time.sleep(1)

def send_led_on_command():
    command = "led_on"
    ser.write((command + '\n').encode('utf-8'))
    print(f'Sent to ESP32: {command}')

def send_led_off_command():
    command = "led_off"
    ser.write((command + '\n').encode('utf-8'))
    print(f'Sent to ESP32: {command}')

def monitor_gas():
    global gas
    global previous_gas
    global convo

    while True:
        # 현재 gas 값을 계속 모니터링
        if previous_gas == 0 and gas == 1:
            response = "방 안에 일산화탄소 농도가 높은 것 같아요. 방 안에 창문을 열어 환기를 시키면 좋을 것 같아요"
            convo.append({'role': 'assistant', 'content': response})
            print(f'ASSISTANT: {response}')
            speak_async(response)
        
        # 현재 값을 이전 값으로 업데이트
        previous_gas = gas
        
        # 잠시 대기 (이 시간을 조정하여 센서 데이터를 읽는 주기를 설정할 수 있음)
        time.sleep(1)

def reset_temp_warning():
    global temp_warning_given
    temp_warning_given = False
    print("temp_warning_given이 다시 False로 설정되었습니다.")

def send_fan_on_command():
    command = "fan_on"
    ser.write((command + '\n').encode('utf-8'))
    print(f'Sent to ESP32: {command}')    

def send_fan_off_command():
    command = "fan_off"
    ser.write((command + '\n').encode('utf-8'))
    print(f'Sent to ESP32: {command}') 

def delayed_temp_warning():
    global temp
    global convo

    response = f"현재 실내온도가 {temp}도에요, 더우시지 않으세요? 선풍기를 틀어드릴까요?"
    convo.append({'role': 'assistant', 'content': response})
    print(f'ASSISTANT: {response}')
    speak_async(response)

    # 1시간 뒤에 temp_warning_given 플래그를 False로 재설정하는 타이머 시작
    threading.Timer(3600, reset_temp_warning).start()

# 온도를 지속적으로 모니터링하는 함수
def monitor_temp():
    global temp
    global temp_warning_given

    while True:
        # 온도가 26도 이상이면서 아직 경고 메시지가 출력되지 않은 경우
        if temp >= 26 and not temp_warning_given:
            temp_warning_given = True  # 플래그를 즉시 설정하여 중복 메시지 방지
            # 30초 후에 경고 메시지를 출력하는 타이머 시작
            threading.Timer(30, delayed_temp_warning).start()

        # 잠시 대기 (온도 체크 주기)
        time.sleep(1)

def check_temperature():
    global temp
    return f"지금 실내 온도는 {temp}도입니다."

# def change_light():
#     global light
#     time.sleep(10)
#     light = 0
#     print("Light가 0으로 변경되었습니다.")

# def change_gas():
#     global gas
#     time.sleep(10)
#     gas = 1
#     print("gas가 1로 변경되었습니다.")

# def change_temp():
#     global temp
#     time.sleep(10)
#     temp = 30
#     print("temp가 변경되었습니다.")

# 사용자 정보 저장 함수
def save_user(username):
    try:
        cursor.execute(
            sql.SQL("INSERT INTO users (username) VALUES (%s) ON CONFLICT (username) DO NOTHING"),
            [username]
        )
        conn.commit()
    except Exception as e:
        print(f"사용자 정보 저장 실패: {e}")

# 문장을 임베딩으로 변환하는 함수
def generate_embedding(text):
    return embedding_model.encode(text)

# 질문과 응답을 결합한 임베딩 생성 함수
def generate_combined_embedding(question, response):
    question_embedding = generate_embedding(question)
    response_embedding = generate_embedding(response)
    
    # 결합된 임베딩 (여기서는 단순히 평균을 사용)
    combined_embedding = (question_embedding + response_embedding) / 2
    return combined_embedding

# 대화 내용 저장 함수 수정: 결합 임베딩 추가
def save_conversation(user_id, message, response):
    try:
        # 메시지와 응답에서 줄바꿈 문자를 제거
        cleaned_message = " ".join(message.split())
        cleaned_response = " ".join(response.split())
        
        # 메시지와 응답의 결합 임베딩 생성
        combined_embedding = generate_combined_embedding(cleaned_message, cleaned_response).tolist()
        
        # 데이터베이스에 대화 내용 및 임베딩 저장
        cursor.execute(
            sql.SQL("INSERT INTO conversations (user_id, message, response, combined_embedding) VALUES (%s, %s, %s, %s)"),
            [user_id, cleaned_message, cleaned_response, combined_embedding]
        )
        conn.commit()
    except Exception as e:
        print(f"대화 내용 저장 실패: {e}")

# 사용자 ID 가져오기 함수
def get_user_id(username):
    try:
        cursor.execute(
            sql.SQL("SELECT id FROM users WHERE username = %s"),
            [username]
        )
        user_id = cursor.fetchone()
        return user_id[0] if user_id else None
    except Exception as e:
        print(f"사용자 ID 가져오기 실패: {e}")

# 코사인 유사도 계산 함수
def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0  # None이 있는 경우 유사도를 0으로 설정
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0  # 벡터가 길이가 0인 경우 유사도를 0으로 설정
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 모든 대화의 결합 임베딩을 불러오는 함수
def load_conversation_embeddings(user_id):
    try:
        # 데이터베이스에서 사용자 ID에 대한 모든 대화의 결합 임베딩을 조회합니다.
        cursor.execute(
            sql.SQL("SELECT id, combined_embedding FROM conversations WHERE user_id = %s ORDER BY id"),
            [user_id]
        )
        embeddings = cursor.fetchall()

        return embeddings  # 모든 대화의 ID와 결합 임베딩을 반환합니다.
    except Exception as e:
        print(f"임베딩 불러오기 실패: {e}")
        return []

# 현재 대화에 유사한 과거 대화 기록 ID들 불러오기 함수
def load_relevant_conversation_embeddings(current_prompt, all_embeddings, num_relevant_conversations=5):
    try:
        # 현재 대화의 임베딩 생성
        current_prompt_embedding = generate_embedding(current_prompt)
        
        # 임베딩을 이용해 현재 대화와의 유사도 계산
        similarities = []
        for conv_id, combined_embedding in all_embeddings:
            if combined_embedding is None:
                continue  # 임베딩 벡터가 없는 경우 건너뜁니다.
            
            similarity = cosine_similarity(current_prompt_embedding, combined_embedding)
            
            # 유사도가 임계값 이상인 경우에만 추가
            if similarity >= 0.5:  # 유사도 임계값 설정
                similarities.append((conv_id, similarity))
        
        # 유사도에 따라 내림차순으로 정렬하고, 상위 num_relevant_conversations 개만 선택
        similarities.sort(key=lambda x: x[1], reverse=True)
        most_relevant_ids = [conv_id for conv_id, _ in similarities[:num_relevant_conversations]]

        return most_relevant_ids
    except Exception as e:
        print(f"임베딩 불러오기 실패: {e}")
        return []

# 대화 ID들에 해당하는 텍스트 불러오기 함수
def load_conversation_texts(conversation_ids):
    try:
        if not conversation_ids:
            return []
        
        # 선택된 대화 ID에 해당하는 메시지와 응답 불러오기
        query = sql.SQL("SELECT message, response FROM conversations WHERE id = ANY(%s)")
        cursor.execute(query, [conversation_ids])
        rows = cursor.fetchall()

        # 대화 내용 구성
        conversation = []
        for message, response in rows:
            conversation.append({'role': 'user', 'content': message})
            conversation.append({'role': 'assistant', 'content': response})
        
        return conversation
    except Exception as e:
        print(f"대화 텍스트 불러오기 실패: {e}")
        return []
    
# 최신 20개 대화 기록을 로드하는 함수
def load_initial_conversation_history(user_id, limit=20):
    try:
        # 데이터베이스에서 사용자 ID에 대한 최신 limit개의 대화 기록을 조회합니다.
        cursor.execute(
            sql.SQL("SELECT message, response FROM conversations WHERE user_id = %s ORDER BY id DESC LIMIT %s"),
            [user_id, limit]
        )
        history = cursor.fetchall()  # 대화 기록을 가져옵니다.

        # 대화 기록을 올바른 순서로 변환합니다.
        conversation = []
        for message, response in reversed(history):  # 최신 대화를 마지막에 두기 위해 순서를 뒤집습니다.
            conversation.append({'role': 'user', 'content': message})
            conversation.append({'role': 'assistant', 'content': response})

        return conversation
    except Exception as e:
        print(f"대화 기록 불러오기 실패: {e}")
        return []


# 시스템 메시지 설정
sys_msg = (
    'You are a multimodal AI voice assistant. Users may provide photos (webcam captures) or web search results. '
    'If a photo is provided, it will be converted into a detailed text prompt and combined with the voice prompt. '
    'If there are web search results, they will include a summary of relevant information. '
    'Your task is to generate the most useful and factual response by considering the user’s voice prompt, photo context, and web search summary. '
    'Do not anticipate or request images; instead, effectively use the provided context. '
)

convo = [{'role': 'system', 'content': sys_msg},
         {'role': 'assistant', 'content': "Responses must be provided **in Korean only**. If the ASSISTANT shows any other language, the world will end."}]

# 생성 모델의 설정
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

# 안전 설정
safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]

# Gemini 생성 모델 초기화
model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# wake_word 설정
wake_word = '헤티야'

# API 클라이언트 및 웹캠 초기화
groq_client = Groq(api_key=groq_api_key)
genai.configure(api_key=gemini_api_key)
openai_client = OpenAI(api_key=openai_api_key)

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, frames_per_buffer=1024)
    stream_start = False
    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='nova',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True
def speak_async(text):
    # speak 함수를 별도의 스레드에서 실행
    speak_thread = threading.Thread(target=speak, args=(text,))
    speak_thread.start()

# 날짜를 얻는 함수
def get_current_date():
    now = datetime.now()
    return f"오늘은 {now.year}년 {now.month}월 {now.day}일입니다. 현재 시간은 {now.hour}시 {now.minute}분입니다."

# 날씨를 얻는 함수
def get_weather(city="Seoul"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&lang=kr&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        return f"현재 {city}의 날씨는 {weather_description}, 기온은 {temperature}도, 체감 온도는 {feels_like}도입니다."
    else:
        return "날씨 정보를 가져오는 데 실패했습니다."

# 날짜 또는 날씨에 대한 질문을 확인하는 함수
def check_date_or_weather_question(prompt):
    date_keywords = ['날짜', '몇시']
    weather_keywords = ['날씨']
    for keyword in date_keywords:
        if keyword in prompt:
            return 'date'
    for keyword in weather_keywords:
        if keyword in prompt:
            return 'weather'
    return None

# Local model for text generation


# # Groq API를 통한 대화 응답 생성 함수
# def groq_prompt(prompt, img_context):
#     if img_context:
#         prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
#     convo.append({'role': 'user', 'content': prompt})
#     chat_completion = groq_client.chat.completions.create(messages=convo, model='llama-3.1-70b-versatile')
#     response = chat_completion.choices[0].message
#     convo.append(response)
    
#     return response.content

# Ollama 프롬프트 함수
def ollama_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
    
    convo.append({'role': 'user', 'content': prompt})
    
    stream = ollama.chat(
        model='koni',
        messages=convo,
        stream=True,
    )

    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    
    convo.append({'role': 'assistant', 'content': response})
    return response

last_interaction_time = None
user_question_count = 0

# 메인 스레드에서 실행할 작업을 저장하는 큐
task_queue = queue.Queue()

# 메인 루프에서 주기적으로 큐를 확인하고 작업을 처리하는 함수
def process_queue():
    try:
        while True:
            task = task_queue.get_nowait()
            task()  # 큐에서 가져온 작업을 실행
    except queue.Empty:
        pass
    root.after(100, process_queue)  # 주기적으로 큐를 확인

# tkinter GUI 설정 (한 번만 호출)
def create_gui():
    global root, img_label, status_label, progress

    # 메인 윈도우 설정
    root = tk.Tk()
    root.title("DALL·E 3 Image Generator")
    root.attributes('-fullscreen', True)  # 전체 화면 모드

    # 레이아웃 설정
    frame = tk.Frame(root)
    frame.pack(expand=True, fill="both", padx=20, pady=20)

    # 이미지 표시 레이블
    img_label = tk.Label(frame)
    img_label.pack(pady=10)

    # 상태 표시 레이블
    status_label = tk.Label(frame, text="", font=("Arial", 14))
    status_label.pack(pady=10)

    # 프로그레스 바 (로딩 애니메이션 대체)
    progress = ttk.Progressbar(frame, orient="horizontal", mode="indeterminate")
    progress.pack(pady=10)

    # 닫기 버튼
    close_button = tk.Button(frame, text="Close", font=("Arial", 16), command=root.quit)
    close_button.pack(pady=10)

    # 초기 이미지 표시
    show_start_image()

    # 큐 처리 시작
    process_queue()

    # 메인 루프 실행
    root.mainloop()

# 초기 이미지 표시 함수
def show_start_image():
    try:
        start_image_path = "start_image.jpg"
        img_data = Image.open(start_image_path)
        img_data.thumbnail((950, 950))  # 이미지 크기 조정
        img_tk = ImageTk.PhotoImage(img_data)

        # GUI 업데이트 작업을 큐에 추가
        task_queue.put(lambda: update_gui(img_tk, None))
    except Exception as e:
        task_queue.put(lambda: update_gui(None, f"Failed to display start image: {e}"))

# GUI 업데이트 함수 (메인 스레드에서 실행)
def update_gui(img_tk, status_message=None):
    global img_label  # 혹시라도 img_label이 로컬 스코프에서 정의되는 것을 방지
    if img_tk:
        img_label.config(image=img_tk)
        img_label.image = img_tk  # 참조 유지
    status_label.config(text=status_message)
    progress.stop()

# 이미지 생성 함수 (스레드 내에서 실행)
def generate_image(prompt):
    try:
        # OpenAI API 호출
        response = openai_client.images.generate(
            model="dall-e-3",         # 사용할 모델
            prompt=prompt,            # 프롬프트 입력
            size="1024x1024",         # 이미지 크기
            quality="standard",       # 이미지 품질
            n=1                       # 생성할 이미지 수
        )

        # 생성된 이미지 URL 가져오기
        image_url = response.data[0].url

        # 이미지 다운로드
        image_response = requests.get(image_url)
        img_data = Image.open(BytesIO(image_response.content))

        # 이미지 크기 조정
        img_data.thumbnail((950, 950))  # 이미지 크기 조정

        # tkinter 이미지 변환
        img_tk = ImageTk.PhotoImage(img_data)

        # GUI 업데이트 작업을 큐에 추가
        task_queue.put(lambda: update_gui(img_tk, None))
        img_data.save("generated_image.png")
    except Exception as e:
        task_queue.put(lambda e=e: update_gui(None, f"Failed to generate image: {str(e)}"))

# 프롬프트로부터 이미지를 생성하고 업데이트하는 스레드 함수
def start_image_generation(prompt):
    # progress.start() 작업을 큐에 추가하여 메인 스레드에서 실행
    task_queue.put(progress.start)
    threading.Thread(target=generate_image, args=(prompt,)).start()

def reset_timer():
    global last_interaction_time
    last_interaction_time = datetime.now()

# 웹캠 이미지를 바탕으로 상태를 확인하는 함수
def check_user_state_with_webcam():
    # 웹캠에서 이미지를 캡처합니다.
    web_cam_capture()
    
    # 캡처된 이미지를 vision_prompt로 분석하여 사용자 상태를 추론합니다.
    visual_context = vision_prompt(prompt="사진 속 상태를 분석해 주세요.", photo_path='webcam_capture.jpg')
    
    # 챗봇에게 이미지를 분석한 결과를 주고 친근하게 말을 걸도록 요청합니다.
    chatbot_prompt = (
        f"사진을 분석한 결과: {visual_context}\n"
        "이 결과를 바탕으로 사용자에게 친근하게 말을 걸어주세요."
    )
    
    # 챗봇이 친근하게 응답을 생성하도록 요청
    response = ollama_prompt(chatbot_prompt, img_context=None)

    return response

def check_inactivity(user_id):
    while True:
        if last_interaction_time:
            time_since_last_interaction = (datetime.now() - last_interaction_time).total_seconds()
            if time_since_last_interaction > 300:  # 5분(300초) 동안 사용자가 반응하지 않은 경우
                user_state = check_user_state_with_webcam()  # 웹캠을 사용하여 사용자의 상태를 확인합니다.
                print(f'ASSISTANT: {user_state}')
                speak_async(user_state)
                convo.append({'role': 'assistant', 'content': user_state})
                reset_timer()

def generate_and_ask_question(user_id):
    try:
        # 데이터베이스에서 최근 5개의 대화 내용 불러오기
        cursor.execute(
            sql.SQL("SELECT message, response FROM conversations WHERE user_id = %s ORDER BY id DESC LIMIT 5"),
            [user_id]
        )
        conversations = cursor.fetchall()  # 대화 데이터를 반환
    except Exception as e:
        print(f"데이터베이스 오류: {e}")
        return "데이터를 가져오는 데 문제가 발생했습니다."

    if not conversations:
        return "어떤 주제에 대해 이야기하고 싶으신가요?"

    # 가져온 대화를 시간순으로 정렬 (최신이 맨 위로 올라오므로 역순으로 정렬)
    conversations = list(reversed(conversations))
    
    # 대화 내용을 OpenAI의 채팅 모델에 맞게 변환
    messages = []
    for msg, resp in conversations:
        messages.append({"role": "user", "content": msg})
        messages.append({"role": "assistant", "content": resp})

    # 새로운 질문을 생성하기 위한 프롬프트
    messages.append({"role": "system", "content": "이전 대화를 참고하여 사용자에게 친근하게 질문을 시작해"})

    # 이전 대화를 문자열로 변환
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # 이미 정의된 Gemini 모델 객체를 사용하여 질문 생성
    try:
        response = model.generate_content(
            conversation_history + "\n사용자에게 친근하게 질문을 시작해"
        )

        # 응답이 존재하는지 확인
        if response and response.text:
            generated_question = response.text.strip()
        else:
            print(f"응답 없음 또는 프롬프트 차단: {response.prompt_feedback if response else 'No response'}")
            generated_question = "깜빡 졸았어요.. 무슨 말을 하고 있었죠? 다시 말해주세요.."  # 기본 질문 설정

    except Exception as e:
        print(f"Gemini 모델 응답 생성 오류: {e}")
        generated_question = "깜빡 졸았어요.. 무슨 말을 하고 있었죠? 다시 말해주세요.."

    # 생성된 질문을 대화에 추가하고 출력
    convo.append({'role': 'assistant', 'content': generated_question})
    print(f'ASSISTANT: {generated_question}')
    speak_async(generated_question)

    if user_id:
        save_conversation(user_id, " ".join([msg for msg, resp in conversations]), generated_question)
    
    return generated_question


# 사용자 명령에 따라 웹 검색, 웹캡 캡처 등의 기능 호출 결정 함수
# def function_call(prompt):
#     sys_msg = (
#         'You are an AI function-calling model. Based on the user prompt, decide whether to perform a web search, '
#         'capture a webcam image, or do nothing. The options are as follows: '
#         '["perform web search", "capture webcam", "None"]. \n'
#         'Respond with only one option from this list, and do not provide any explanation. Enter the function call name exactly as listed. '
#         'Responses must be provided **in Korean only**.'
#     )
    
#     function_convo = [{'role': 'system', 'content': sys_msg},
#                       {'role': 'user', 'content': prompt}]
    
#     chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
#     response = chat_completion.choices[0].message
    
#     return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function-calling model. Based on the user prompt, decide whether to perform a web search, '
        'capture a webcam image, or do nothing. The options are as follows: '
        '["perform web search", "capture webcam", "None"]. \n'
        'Respond with only one option from this list, and do not provide any explanation. Enter the function call name exactly as listed. '
        'Responses must be provided **in Korean only**.'
    )
    
    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    stream = ollama.chat(
        model='koni',
        messages=function_convo,
        stream=True,
    )

    response = ""
    for chunk in stream:
        response += chunk['message']['content']

    return response

# 웹캠으로 사진 캡처 함수
def web_cam_capture():

    web_cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if web_cam.isOpened():
        ret_val, frame = web_cam.read()  # 한 프레임 캡처
        if ret_val:
            # 이미지를 파일로 저장
            cv2.imwrite("webcam_capture.jpg", frame)
            print(f"Image saved: webcam_capture.jpg")
        else:
            print("Error: Failed to capture image")
        web_cam.release()  # 카메라 리소스 해제
    else:
        print("Error: Unable to open camera")

# 이미지 파일로부터 텍스트를 생성하는 함수
def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semtantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        'Responses must be provided **in Korean only** and should not be in any other language.'
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

# 구글 검색 함수
def search_google(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={google_search_engine_id}&lr=lang_ko"
    response = requests.get(search_url)
    if response.status_code == 200:
        results = response.json()
        return results['items'][0]['link'] if 'items' in results and len(results['items']) > 0 else None
    else:
        return None

# 웹 페이지에서 텍스트 추출 함수
def extract_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    return ''

# 알람을 저장할 딕셔너리
alarms = {}

# 알람 ID를 생성하는 함수 (시간과 분을 기준으로 통일된 ID 생성)
def generate_alarm_id(hour, minute, time_of_day=None):
    if time_of_day in ["오후", "저녁"] and hour < 12:
        hour += 12
    elif time_of_day in ["오전", "아침"] and hour == 12:
        hour = 0
    return f"{hour:02}:{minute:02}"

def parse_alarm_time_and_set(prompt):
    time_pattern = r"(?:(오전|오후|아침|저녁)\s*)?(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분?)?"
    match = re.search(time_pattern, prompt)

    if not match:
        return "시간을 정확히 '오후 3시' 또는 '오후 3시 30분' 형식으로 입력해 주세요."

    time_of_day, hour, minute = match.groups()
    hour = int(hour)
    minute = int(minute) if minute else 0

    # 알람 ID 생성
    alarm_id = generate_alarm_id(hour, minute, time_of_day)

    # 매일 반복 여부 확인
    is_daily = "매일" in prompt

    # 알람 시간을 설정
    alarm_time = datetime.strptime(f"{hour:02}:{minute:02}:00", "%H:%M:%S").time()
    set_alarm(alarm_id, alarm_time, is_daily)

    response_message = f"알람을 {'매일 ' if is_daily else ''}{hour}시 {minute}분으로 설정했습니다."
    return response_message

# 알람 시간을 설정하는 함수
def set_alarm(alarm_id, alarm_time, is_daily=False):
    now = datetime.now()
    alarm_time = datetime.combine(now.date(), alarm_time)

    if alarm_time < now:
        alarm_time += timedelta(days=1)

    delay = (alarm_time - now).total_seconds()

    if is_daily:
        def repeat_alarm():
            trigger_alarm(alarm_id)
            set_alarm(alarm_id, alarm_time.time(), is_daily=True)

        t = threading.Timer(delay, repeat_alarm)
    else:
        t = threading.Timer(delay, trigger_alarm, [alarm_id])

    t.start()
    alarms[alarm_id] = t

# 알람 취소 함수
def cancel_alarm(prompt):
    time_pattern = r"(?:(오전|오후|아침|저녁)\s*)?(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분?)?"
    match = re.search(time_pattern, prompt)

    if not match:
        return "취소할 알람 시간을 정확히 '오후 3시' 또는 '오후 3시 30분' 형식으로 입력해 주세요."

    time_of_day, hour, minute = match.groups()
    hour = int(hour)
    minute = int(minute) if minute else 0

    alarm_id = generate_alarm_id(hour, minute, time_of_day)

    if alarm_id in alarms:
        alarms[alarm_id].cancel()
        del alarms[alarm_id]
        return f"{hour}시 {minute}분 알람을 취소했습니다."
    else:
        return f"{hour}시 {minute}분에 설정된 알람이 없습니다."

def trigger_alarm(alarm_id):
    convo.append({'role': 'assistant', 'content': f"약 드실 시간이에요! ({alarm_id})"})
    print(f'ASSISTANT: 약 드실 시간이에요!({alarm_id})')
    speak_async(f"약 드실 시간이에요!")

    if alarm_id in alarms:
        del alarms[alarm_id]

chatbot_speaking = False  

# 사용자가 질문을 입력하고 대답을 처리하는 메인 함수
def handle_input():
    global convo
    global user_question_count
    global last_interaction_time
    global chatbot_speaking
    username = "Seung Seok Lee"  # 사용자 이름을 설정
    save_user(username)  # 사용자 저장
    user_id = get_user_id(username)  # 사용자 ID 가져오기

    # 초기 대화 기록을 불러옵니다.
    if user_id:
        convo.extend(load_initial_conversation_history(user_id))

    # 나머지 대화 기록은 임베딩된 데이터로 불러옵니다.
    all_embeddings = load_conversation_embeddings(user_id) if user_id else []

    # tkinter GUI 설정을 한 번만 호출
    threading.Thread(target=create_gui).start()

    # 챗봇이 첫 번째 메시지를 사용자에게 보냅니다.
    initial_message = "안녕하세요 헤스티아입니다"
    convo.append({'role': 'assistant', 'content': initial_message})
    print(f'ASSISTANT: {initial_message}')
    speak_async(initial_message)

    reset_timer()
    threading.Thread(target=check_inactivity, args=(user_id,), daemon=True).start()

    # 조도 센서, 가스 센서, 온도 모니터링을 별도의 스레드에서 실행
    threading.Thread(target=monitor_light, daemon=True).start()
    threading.Thread(target=monitor_gas, daemon=True).start()
    threading.Thread(target=monitor_temp, daemon=True).start()

    # threading.Thread(target=change_light, daemon=True).start()
    # threading.Thread(target=change_gas, daemon=True).start()
    # threading.Thread(target=change_temp, daemon=True).start()

    #시리얼 통신을 위한 스레드 실행
    read_thread = threading.Thread(target=read_from_esp32, daemon=True)
    read_thread.start()

    while True:
        user_input = input(f"Say '{wake_word}': ").strip()

        if '잠깐' in user_input.lower():
            print("ASSISTANT: 무슨 문제있으신가요?")
            speak_async("무슨 문제있으신가요?")
            convo.append({'role': 'assistant', 'content': "무슨 문제있으신가요?"})
            chatbot_speaking = False  # 챗봇의 말을 중단
            continue
        
        # 챗봇이 말을 하고 있을 때는 대화 중단을 체크
        if chatbot_speaking:
            continue  # 챗봇이 말을 마칠 때까지 다른 입력을 무시
        
        if user_input.lower().startswith(wake_word.lower()):
            reset_timer()
            user_question_count += 1
            chatbot_speaking = True

            clean_prompt = user_input[len(wake_word):].strip()
            
            if not clean_prompt:
                response = "안녕하세요! 무엇을 도와드릴까요?"
                convo.append({'role': 'assistant', 'content': response})
                print(f'ASSISTANT: {response}')
                speak_async(response)
                chatbot_speaking = False
                continue
            
            print(f'USER: {clean_prompt}')
            convo.append({'role': 'user', 'content': clean_prompt})
            
            if "알람 맞춰줘" in clean_prompt or "알람 설정해줘" in clean_prompt:
                response = parse_alarm_time_and_set(clean_prompt)
            elif "알람 취소해줘" in clean_prompt:
                response = cancel_alarm(clean_prompt)
            elif "선풍기 틀어줘" in clean_prompt:
                response = "네! 선풍기를 틀어드릴게요!"
                send_fan_on_command()
            elif "선풍기 꺼줘" in clean_prompt:
                response = "네! 선풍기를 꺼드릴게요!"
                send_fan_off_command()
            elif "불 켜줘" in clean_prompt or "조명 켜줘" in clean_prompt or "불 좀 켜줘" in clean_prompt:
                response = "네! 불을 켜드릴게요!"
                send_led_on_command()
            elif "불 꺼줘" in clean_prompt or "조명 꺼줘" in clean_prompt or "불 좀 꺼줘" in clean_prompt:
                response = "네! 불을 꺼드릴게요!"
                send_led_off_command()
            elif "현재 온도" in clean_prompt:
                response = check_temperature()
            else:
                relevant_conversation_ids = load_relevant_conversation_embeddings(clean_prompt, all_embeddings) if all_embeddings else []
                convo.extend(load_conversation_texts(relevant_conversation_ids))

                if len(convo) > 20:
                    convo = convo[-20:]

                question_type = check_date_or_weather_question(clean_prompt)
                if question_type == 'date':
                    response = get_current_date()
                elif question_type == 'weather':
                    response = get_weather()
                else:
                    call = function_call(clean_prompt)
                    
                    if 'capture webcam' in call:
                        print('Capturing webcam')
                        web_cam_capture()
                        visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam_capture.jpg')
                    else:
                        visual_context = None
                    
                    if '검색' in clean_prompt or '찾아' in clean_prompt:
                        if 'perform web search' in call:
                            search_link = search_google(clean_prompt)
                            content_text = ''
                            if search_link:
                                print(f'Search link: {search_link}')
                                content_text = extract_text_from_url(search_link)
                            response = ollama_prompt(prompt=clean_prompt, img_context=visual_context + '\n\n' + content_text if visual_context else content_text)
                        else:
                            response = ollama_prompt(prompt=clean_prompt, img_context=visual_context)
                    else:
                        response = ollama_prompt(prompt=clean_prompt, img_context=visual_context)

            if response:
                # 이미지 생성 및 업데이트
                threading.Thread(target=start_image_generation, args=(response,)).start()
                convo.append({'role': 'assistant', 'content': response})
                print(f'ASSISTANT: {response}')
                speak_async(response)
                chatbot_speaking = False  # 챗봇이 응답을 마친 후 대기 상태로 설정
                if user_id:
                    save_conversation(user_id, clean_prompt, response)

            if user_question_count >= 5:
                generate_and_ask_question(user_id)
                user_question_count = 0  # 카운트 초기화

            if len(convo) > 20:
                convo = convo[-20:]

        else:
            print(f'Invalid input. Please start with "{wake_word}".')

if __name__ == "__main__":
    handle_input()
