"""
Модуль регистрации событий
Класс: Logger
"""

import json
import csv
import xml.etree.ElementTree as ET
import os
from datetime import datetime
from typing import Dict, Any, List


class Logger:
    """Регистрация событий в электронном журнале (CSV, JSON, XML)"""

    def __init__(self, log_file_path: str = "gesture_log", session_id: str = None, format: str = "json"):
        """
        Инициализация логгера

        Args:
            log_file_path: путь к файлу журнала (без расширения)
            session_id: идентификатор сессии
            format: формат выходного файла (csv, json, xml)
        """
        self.log_file_path = log_file_path
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.format = format.lower()
        self.records: List[Dict[str, Any]] = []

        # Создание папки для логов
        os.makedirs("logs", exist_ok=True)

    def log_event(self, performer_name: str, gesture_type: str, confidence: float,
                  is_abnormal: bool = False, abnormal_reason: str = "") -> bool:
        """Добавление записи в журнал"""
        record = {
            "id": len(self.records) + 1,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "performer_name": performer_name,
            "gesture_type": gesture_type if gesture_type else "unknown",
            "confidence": round(confidence, 2),
            "is_abnormal": 1 if is_abnormal else 0,
            "abnormal_reason": abnormal_reason if is_abnormal else ""
        }
        self.records.append(record)

        # Немедленная запись на диск
        if self.format == "csv":
            self._save_to_csv(record)
        elif self.format == "json":
            self._save_to_json(record)
        elif self.format == "xml":
            self._save_to_xml(record)

        return True

    def _save_to_csv(self, record: Dict[str, Any]):
        """Сохранение записи в CSV"""
        file_path = f"logs/{self.log_file_path}_{self.session_id}.csv"
        file_exists = os.path.exists(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def _save_to_json(self, record: Dict[str, Any]):
        """Сохранение в JSON"""
        file_path = f"logs/{self.log_file_path}_{self.session_id}.json"

        existing_records = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_records = json.load(file)

        existing_records.append(record)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(existing_records, file, ensure_ascii=False, indent=2)

    def _save_to_xml(self, record: Dict[str, Any]):
        """Сохранение в XML"""
        file_path = f"logs/{self.log_file_path}_{self.session_id}.xml"

        if os.path.exists(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()
        else:
            root = ET.Element("gesture_log")
            root.set("session_id", self.session_id)
            root.set("start_time", datetime.now().isoformat())
            tree = ET.ElementTree(root)

        record_elem = ET.SubElement(root, "record")
        for key, value in record.items():
            field = ET.SubElement(record_elem, key)
            field.text = str(value)

        tree.write(file_path, encoding='utf-8', xml_declaration=True)

    def export_log(self, export_format: str = "json") -> str:
        """Экспорт всего журнала"""
        if not self.records:
            return "Журнал пуст"

        output_file = f"logs/export_{self.session_id}.{export_format}"

        if export_format == "csv":
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.records[0].keys())
                writer.writeheader()
                writer.writerows(self.records)
        elif export_format == "json":
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(self.records, file, ensure_ascii=False, indent=2)
        elif export_format == "xml":
            root = ET.Element("gesture_log")
            root.set("session_id", self.session_id)
            for record in self.records:
                record_elem = ET.SubElement(root, "record")
                for key, value in record.items():
                    field = ET.SubElement(record_elem, key)
                    field.text = str(value)
            tree = ET.ElementTree(root)
            tree.write(output_file, encoding='utf-8', xml_declaration=True)

        return f"Журнал экспортирован в {output_file}"