import cv2
import os
import numpy as np
from collections import Counter
from fer.fer import FER

try:
    # Utilizamos a lib ultralytics que nos fornece o modelo YOLOv8, sendo ideal para detecção de objetos/atividades com base no nosso vídeo
    from ultralytics import YOLO
    
    # Carregamos um modelo pré-treinado (modelo 'nano', que é o mais leve, para uso mais rápido e sem consumir muito processamento da CPU)
    YOLO_MODEL_PATH = 'yolov8n.pt'
    activity_model = YOLO(YOLO_MODEL_PATH)
    ACTIVITY_DETECTION_AVAILABLE = True
    
    # Mapeamos alguns objetos para detectar atividades humanas mais simples, além de facilitar a leitura do resumo final
    # Em inglês os nomes dos objetos detectados pelo modelo YOLOv8 (com base na documentação oficial do modelo),
    # Traduzimos as atividades para português, para melhor entendimento da análise final
    ACTIVITY_MAP = {
        'laptop': 'Trabalhando/Estudando',
        'cell phone': 'Comunicando',
        'book': 'Lendo',
        'sports ball': 'Esporte/Lazer',
        'cup': 'Consumindo Bebida',
        'mouse': 'Trabalhando/Estudando',
        'keyboard': 'Trabalhando/Estudando',
        'tv': 'Assistindo',
        'chair': 'Sentado',
    }
    
except ImportError:
    # Aqui tratamos o caso onde a biblioteca ultralytics não está instalada
    ACTIVITY_DETECTION_AVAILABLE = False
    print("AVISO: A biblioteca 'ultralytics' não foi encontrada. Para detecção de atividades, é necessário instalá-la.")
    print("Instale com 'pip install ultralytics'.")

emotion_detector = FER(mtcnn=True)
FRAME_SKIP_EMOTION = 5 # Analisa emoção a cada 5 quadros
FRAME_SKIP_ACTIVITY = 10 # Analisa atividade a cada 10 quadros (é mais pesado processar)


def analyze_video_data(video_path="Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"):
    """
    Executa a detecção de faces, análise de emoções e detecção de atividades.
    """
    
    # Configuramos o salvamento das imagens em uma pasta específica
    output_folder = "rostos_detectados_e_atividades"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    # Lista para armazenar dados para o resumo final
    summary_data = []
    frame_count = 0
    face_count_saved = 0
    
    # Obtém o FPS do vídeo para calcular o tempo total de vídeo processado
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    print("Iniciando processamento do vídeo...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_frame_activities = []
            
            # Aqui detectamos emoções a cada N quadros (15, conforme FRAME_SKIP_EMOTION definido anteriormente)
            current_frame_emotions = []
            if frame_count % FRAME_SKIP_EMOTION == 0:
                results = emotion_detector.detect_emotions(frame)
                
                for face_info in results:
                    (x, y, w, h) = face_info["box"]
                    emotions = face_info["emotions"]
                    dominant_emotion = max(emotions, key=emotions.get)
                    score = emotions[dominant_emotion]
                    
                    current_frame_emotions.append(dominant_emotion)

                    # Desenha a caixa para Reconhecimento Facial
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Adiciona o texto da Análise de Expressões Emocionais
                    label = f"{dominant_emotion}: {score:.2f}"
                    cv2.putText(
                        frame, 
                        label, 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )

                    # Salva a imagem do rosto apenas neste quadro analisado
                    if frame_count % FRAME_SKIP_ACTIVITY == 0:
                        face_img = frame[y : y + h, x : x + w]
                        # Adiciona a emoção e o quadro ao nome do arquivo
                        face_filename = os.path.join(
                            output_folder, 
                            f"frame{frame_count}_face{face_count_saved}_{dominant_emotion}.jpg"
                        )
                        cv2.imwrite(face_filename, face_img)
                        face_count_saved += 1

            # Aqui detectamos atividades a cada N quadros (10, conforme FRAME_SKIP_ACTIVITY definido anteriormente)
            if ACTIVITY_DETECTION_AVAILABLE and frame_count % FRAME_SKIP_ACTIVITY == 0:
                
                # Mapeamos os nomes de classes para seus índices numéricos, afim de evitar erros de passagem de parâmetros
                desired_classes_names = ACTIVITY_MAP.keys()
                
                # Criamos um dicionário reverso para lookup rápido: {nome: índice}, ex: {'laptop': 56, ...}
                yolo_names_to_indices = {name: index for index, name in activity_model.names.items()}
                
                # Filtramos apenas os índices que queremos, baseados nos nomes desejados
                desired_classes_indices = [
                    yolo_names_to_indices[name] 
                    for name in desired_classes_names 
                    if name in yolo_names_to_indices
                ]
                
                # O método predict agora recebe a lista de índices (inteiros)
                yolo_results = activity_model.predict(
                    frame, 
                    verbose=False, 
                    conf=0.5, 
                    classes=desired_classes_indices
                )

                for res in yolo_results:
                    for box in res.boxes:
                        # Obtém a classe detectada (ex: 'laptop', 'cell phone')
                        class_id = int(box.cls[0])
                        class_name = activity_model.names[class_id]
                        
                        # Converte a detecção para uma atividade, caso ela esteja no nosso mapeamento
                        if class_name in ACTIVITY_MAP:
                            activity = ACTIVITY_MAP[class_name]
                            if activity not in current_frame_activities:
                                current_frame_activities.append(activity)
                            
                            # Desenha a caixa e o rótulo da Atividade
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(
                                frame, 
                                activity, 
                                (x1, y1 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (255, 0, 255), 
                                2
                            )

            # Aqui armazenamos os dados deste quadro para o resumo final
            if current_frame_emotions or current_frame_activities:
                summary_data.append({
                    "frame": frame_count,
                    "time_sec": round(frame_count / fps, 1),
                    "emotions": current_frame_emotions,
                    "activities": current_frame_activities,
                    "face_count": len(current_frame_emotions)
                })

            # Exibição do vídeo
            cv2.imshow("Analise Facial, Emocional e de Atividades", frame)
            
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
                
    except Exception as e:
        print(f"Ocorreu um erro durante o processamento: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Processamento finalizado.")
        
        # Gera o resumo final da análise do vídeo
        generate_summary(summary_data, fps)


def generate_summary(summary_data, fps):
    """
    Gera e imprime o resumo final.
    """
    if not summary_data:
        print("\n--- RESUMO DA ANÁLISE ---")
        print("Nenhum rosto ou atividade detectada nos quadros analisados.")
        return

    # Agregação de dados para o resumo
    total_frames_inferred = len(summary_data)
    total_time_inferred = round(total_frames_inferred * (FRAME_SKIP_EMOTION / fps), 1) # Estimativa de tempo total analisado
    
    all_emotions = [e for item in summary_data for e in item['emotions']]
    all_activities = [a for item in summary_data for a in item['activities']]

    emotion_counts = Counter(all_emotions)
    activity_counts = Counter(all_activities)
    
    total_emotion_detections = len(all_emotions)
    total_activity_detections = len(all_activities)

    # Definimos o nome do arquivo de saída do resumo final
    output_filename = "analise_final.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        
        # Função para imprimir e exportar o resumo final simultaneamente
        def print_and_write(text=""):
            print(text)
            f.write(text + '\n')

        # Impressão e Exportação do resumo final
        print_and_write("\n==============================================")
        print_and_write("           RESUMO DA ANÁLISE DO VÍDEO")
        print_and_write("==============================================")
        print_and_write(f"-> Total de pontos de dados analisados (quadros): {total_frames_inferred}")
        print_and_write(f"-> Duração total detectada (aprox.): {total_time_inferred} segundos")
        print_and_write(f"-> Imagens de rostos salvas na pasta '{os.path.basename(os.getcwd())}/rostos_detectados_e_atividades'.")
        print_and_write("----------------------------------------------")
        
        # Resumo das Emoções Detectadas
        print_and_write("### EMOÇÕES DOMINANTES DETECTADAS")
        print_and_write(f"Total de detecções emocionais: {total_emotion_detections}")
        
        if total_emotion_detections > 0:
            for emotion, count in emotion_counts.most_common():
                percentage = (count / total_emotion_detections) * 100
                print_and_write(f" - {emotion.capitalize()}: {count} detecções ({percentage:.1f}%)")
        else:
            print_and_write("Nenhuma emoção detectada.")
            
        print_and_write("----------------------------------------------")
            
        # Resumo das Atividades Detectadas
        print_and_write("### ATIVIDADES DETECTADAS")
        print_and_write(f"Total de atividades detectadas: {total_activity_detections}")
        
        if total_activity_detections > 0:
            for activity, count in activity_counts.most_common():
                percentage = (count / total_activity_detections) * 100
                print_and_write(f" - {activity}: {count} detecções ({percentage:.1f}%)")
        else:
            print_and_write("Nenhuma atividade detectada (verifique se o modelo YOLO está carregado, se instalação da 'ultralytics' foi feita e se os objetos foram detectados).")

        print_and_write("==============================================")
        print_and_write(f"Resumo da análise final salvo em: '{output_filename}'.")

# Inicia a análise do vídeo
analyze_video_data()