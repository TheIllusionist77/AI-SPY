import pygame, sys, cv2, random, time
from pygame.locals import *
from ultralytics import YOLO

pygame.init()
model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

SCREEN = pygame.display.set_mode((384, 480))

font = "Courier Prime Bold.ttf"
TINIER_FONT = pygame.font.Font(font, 12)
TINY_FONT = pygame.font.Font(font, 16)
SMALL_FONT = pygame.font.Font(font, 24)
FONT = pygame.font.Font(font, 28)
BIG_FONT = pygame.font.Font(font, 36)

household_items = [
    "person", "backpack", "bottle", "cup", "fork", "knife", 
    "spoon", "bowl", "banana", "apple", "orange", "chair", 
    "couch", "bed", "dining table", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", 
    "sink", "refrigerator", "book", "clock", "toothbrush"
]

def detect_objects(frame, image_size, game_win):
    results = model(frame, imgsz = image_size)
    annotated_frame = results[0].plot()
    
    item_list = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            class_name = model.names[class_id]
            
            detection = {"Class": class_name, "Confidence": conf, "BBox": xyxy}
            if conf >= 0.5 or game_win == True:
                item_list.append(detection)
            
    return annotated_frame, item_list

def new_choice():
    choice = random.choice(household_items)
    return choice

choice = new_choice()

item_found = False
show_boxes = False
game_start = False
game_win = False

frame_counter = 0
winning_bbox = 0

while True:
    frame_counter += 1
    start = time.perf_counter()
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                if not game_start or game_win:
                    game_start = True
                    game_win = False
                    choice = new_choice()
                else:
                    show_boxes = not show_boxes
            if event.key == K_s:
                if game_start and not game_win:
                    choice = new_choice()
            if event.key == K_q:
                pygame.quit()
                sys.exit()
            
    result, image = camera.read()
    
    if result:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[0:480, 128:512]
        annotated_frame, objects = detect_objects(image, 480, game_win)
                
        for item in objects:
            if item["Class"] == choice:
                winning_bbox = item["BBox"]
                game_win = True
    
    if not game_start:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 0)
        surface = pygame.surfarray.make_surface(image)
        
        SCREEN.blit(surface, (0, 0))
        
        target = BIG_FONT.render("AI-SPY", False, (0, 255, 0))
        SCREEN.blit(target, (130, 220))
        
        target = TINY_FONT.render("Click space to start", False, (255, 0, 0))
        SCREEN.blit(target, (100, 450))
    elif game_win:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 0)
        surface = pygame.surfarray.make_surface(image)
        
        SCREEN.blit(surface, (0, 0))
        
        pygame.draw.rect(SCREEN, (0, 0, 255), (winning_bbox[0], winning_bbox[1], winning_bbox[2] - winning_bbox[0], winning_bbox[3] - winning_bbox[1]), 3)
        
        target = FONT.render("You found the object!", False, (0, 255, 0))
        SCREEN.blit(target, (20, 220))
        
        target = TINY_FONT.render("Click space to continue", False, (255, 0, 0))
        SCREEN.blit(target, (80, 450))
    elif show_boxes:
        annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        annotated_frame = cv2.flip(annotated_frame, 0)
        surface = pygame.surfarray.make_surface(annotated_frame)
        
        SCREEN.blit(surface, (0, 0))
        
        target = TINY_FONT.render("Click space to switch to list view", False, (255, 0, 0))
        SCREEN.blit(target, (20, 450))
    else:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 0)
        surface = pygame.surfarray.make_surface(image)
        
        SCREEN.blit(surface, (0, 0))
        
        statement = TINY_FONT.render("(conf >= 0.5)", False, (255, 0, 0))
        SCREEN.blit(statement, (20, 280))
        
        statement = FONT.render("I see a...", False, (255, 0, 0))
        SCREEN.blit(statement, (20, 300))
        
        objects.sort(key = lambda x: round(x["Confidence"], 1), reverse = True)
       
        count = 0
        item_list = []
        
        for item in objects:
            if item["Class"] not in item_list and count <= 4:
                item_list.append(item["Class"])
                count += 1
        
        count = 0
        for item in item_list:
            vision = SMALL_FONT.render(item, False, (255, 0, 0))
            SCREEN.blit(vision, (20, 330 + 30 * count))
            
            confidence = TINY_FONT.render("(" + str(round(objects[count]["Confidence"], 1)) + ")", False, (255, 0, 0))
            SCREEN.blit(confidence, (20 + len(item) * 15, 333 + 30 * count))
            count += 1
            
        target = TINY_FONT.render("Click space to switch to box view", False, (255, 0, 0))
        SCREEN.blit(target, (25, 450))
        
    end = time.perf_counter()
    
    total_time = end - start
    FPS = 1 / total_time
        
    frame_rate = TINIER_FONT.render("FPS: " + str(round(FPS)), False, (0, 255, 0))
    SCREEN.blit(frame_rate, (320, 10))
        
    leave = TINIER_FONT.render("Q - quit", False, (255, 0, 0))
    SCREEN.blit(leave, (320, 24))
    
    if game_start:
        prompt = FONT.render("Can you find a...", False, (255, 0, 0))
        SCREEN.blit(prompt, (20, 20))
        
        target = FONT.render(choice + "?", False, (255, 0, 0))
        SCREEN.blit(target, (20, 50))
        
        if not game_win:
            skip = TINIER_FONT.render("S - skip", False, (255, 0, 0))
            SCREEN.blit(skip, (320, 38))
            
    pygame.display.update()