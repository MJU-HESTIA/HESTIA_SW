from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from PIL import Image,ImageGrab
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
# import pyperclip
import pyaudio
from faster_whisper import WhisperModel
import speech_recognition as sr

temp = 25
gas = 30

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

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
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
    'Responses must be provided **in Korean only** and should not be in any other language. '
    'Additionally, you are a warm, friendly, and caring AI assistant designed to converse with elderly individuals living alone. '
    'Your primary goal is to make them feel accompanied, cared for, and provide them with useful information in a compassionate manner. '
    'Always use polite, kind, and encouraging language, making the conversation feel personal and comforting. '
    'When providing instructions or information, ensure that it is simple, clear, and easy to follow. '
    'If the user asks for help, guidance, or just wants to talk, respond with warmth and patience. '
    'Your responses should be sensitive to the user’s feelings and needs. '
    'Use a casual, friendly, and conversational tone, similar to how you would speak to a close friend or family member. '
    'Do not use the formal endings "입니다" or "습니다" in your responses. These expressions are **banned** from your responses. '
    'Instead, always use more conversational endings like "해요" or "드릴게요". '
    'For example, instead of saying "추천해드리겠습니다", you should say "추천해드릴게요". '
    'Ensure that no formal language is used, and focus on making your tone as warm and approachable as possible.'
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
wake_word = '안녕'

# API 클라이언트 및 웹캠 초기화
groq_client = Groq(api_key=groq_api_key)
genai.configure(api_key=gemini_api_key)
openai_client = OpenAI(api_key=openai_api_key)
web_cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

r = sr.Recognizer()
r.energy_threshold = 300
r.dynamic_energy_threshold = False
r.pause_threshold = 0.8
r.language = 'ko-KR'
source = sr.Microphone()

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='alloy',
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

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''. join(segment.text for segment in segments)
    return text

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


# Groq API를 통한 대화 응답 생성 함수
def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama-3.1-70b-versatile')
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content

last_interaction_time = None
user_question_count = 0

def reset_timer():
    global last_interaction_time
    last_interaction_time = datetime.now()

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
    response = groq_prompt(chatbot_prompt, img_context=None)

    return response

def check_inactivity(user_id):
    while True:
        if last_interaction_time:
            time_since_last_interaction = (datetime.now() - last_interaction_time).total_seconds()
            if time_since_last_interaction > 300:  # 5분(300초) 동안 사용자가 반응하지 않은 경우
                user_state = check_user_state_with_webcam()  # 웹캠을 사용하여 사용자의 상태를 확인합니다.
                print(f'ASSISTANT: {user_state}')
                speak(user_state)
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

    # GPT-4o 모델을 사용하여 질문 생성
    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-05-13",  # gpt-4o 모델 사용
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    generated_question = response.choices[0].message.content.strip()

    # 생성된 질문을 대화에 추가하고 출력
    convo.append({'role': 'assistant', 'content': generated_question})
    print(f'ASSISTANT: {generated_question}')
    speak(generated_question)

    if user_id:
        save_conversation(user_id, " ".join([msg for msg, resp in conversations]), generated_question)
    
    return generated_question

def set_alarm(alarm_time):
    now = datetime.now()
    alarm_time = datetime.combine(now.date(), alarm_time)
    
    if alarm_time < now:
        alarm_time += timedelta(days=1)
    
    delay = (alarm_time - now).total_seconds()
    t = threading.Timer(delay, trigger_alarm)
    t.start()

def trigger_alarm():
    convo.append({'role': 'assistant', 'content': "약 드실 시간입니다!"})
    print(f'ASSISTANT: 약 드실 시간입니다!')
    speak("약 드실 시간입니다!")


# 사용자 명령에 따라 웹 검색, 웹캡 캡처 등의 기능 호출 결정 함수
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
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    
    return response.content

# 웹캠으로 사진 캡처 함수
def web_cam_capture():
    if not web_cam.isOpened():
        print("Cannot open camera")
        speak("Cannot open camera")
        exit()
        
    path = 'webcam_capture.jpg'
    ret, frame = web_cam.read()
    if ret:
        cv2.imwrite(path, frame)
    else:
        print("Failed to capture image")
        speak("Failed to capture image")

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

def ir_send():
    print("보냈습니다.")

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

chatbot_speaking = False  

# 사용자가 질문을 입력하고 대답을 처리하는 메인 함수
def callback(recognizer, audio):
    global convo
    global user_question_count
    global last_interaction_time
    global chatbot_speaking
    username = "User3"  # 사용자 이름을 설정
    save_user(username)  # 사용자 저장
    user_id = get_user_id(username)  # 사용자 ID 가져오기

    # 초기 대화 기록을 불러옵니다.
    if user_id:
        convo.extend(load_initial_conversation_history(user_id))

    # 나머지 대화 기록은 임베딩된 데이터로 불러옵니다.
    all_embeddings = load_conversation_embeddings(user_id) if user_id else []

    reset_timer()
    threading.Thread(target=check_inactivity, args=(user_id,), daemon=True).start()

    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
        
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        # "잠깐"이라는 말이 있을 경우 챗봇의 말을 중단하고 응답
        if '잠깐' or '잠깐만' in clean_prompt.lower():
            print("ASSISTANT: 무슨 문제있으신가요?")
            convo.append({'role': 'assistant', 'content': "무슨 문제있으신가요?"})
            chatbot_speaking = False  # 챗봇의 말을 중단
            return

        reset_timer()
        user_question_count += 1
        chatbot_speaking = True
        
        print(f'USER: {clean_prompt}')
        convo.append({'role': 'user', 'content': clean_prompt})

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
                    response = groq_prompt(prompt=clean_prompt, img_context=visual_context + '\n\n' + content_text if visual_context else content_text)
                else:
                    response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
            else:
                response = groq_prompt(prompt=clean_prompt, img_context=visual_context)

        if response:
            convo.append({'role': 'assistant', 'content': response})
            print(f'ASSISTANT: {response}')
            speak(response)
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

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'followed with your prompt. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([가-힣A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

alarm_time = datetime.strptime("20:10:00", "%H:%M:%S").time()
set_alarm(alarm_time)

start_listening()
