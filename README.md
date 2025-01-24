# Классификация Аудио с Использованием Deep Learning

## Описание
Этот проект представляет собой систему классификации аудио на основе глубокого обучения, использующую предобученную модель Wav2Vec2 и трансформеры.

## Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/audio-classification.git
cd audio-classification
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Структура проекта
```
audio-classification/
├── README.md
├── requirements.txt
├── config.yml
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── finetune.py
│   └── utils/
├── data/
└── models/
```

## Использование

### Обучение модели
```bash
python src/train.py --config config.yml
```

### Инференс
```bash
python src/inference.py --audio path/to/audio.wav --model path/to/model.pth
```

### Дообучение модели
```bash
python src/finetune.py --model path/to/model.pth --train_data path/to/train --valid_data path/to/valid
```

## Конфигурация
Все параметры можно настроить в файле `config.yml`:
- Параметры датасета
- Параметры аудио
- Параметры модели
- Параметры обучения

## Требования к данным
- Аудиофайлы должны быть в формате WAV
- Частота дискретизации: 16000 Hz
- Минимальная длительность: 1 секунда

## Результаты
После обучения модели в директории `models/` сохраняются:
- Веса лучшей модели
- Чекпоинты
- Графики обучения
- Матрица ошибок

## Поддержка
По всем вопросам обращайтесь в Issues на GitHub.
