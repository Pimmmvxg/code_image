import json
import time
import paho.mqtt.client as mqtt

THINGSBOARD_HOST = "191.20.110.47"  # หรือ IP Server 
ACCESS_TOKEN = "image_token"  # Token จากอุปกรณ์ใน ThingsBoard
PORT = 1883

class ThingsboardClient:
    def __init__(self, host=THINGSBOARD_HOST, token=ACCESS_TOKEN, port=PORT):
        self.client = mqtt.Client()
        self.client.username_pw_set(token)
        self.client.connect(host, port, 60)
        self.client.loop_start()

    def publish_attributes(self, attributes: dict):
        self.client.publish("v1/devices/me/attributes", json.dumps(attributes), qos=1)
        print("Published attributes:", attributes)
        
    def publish_telemetry(self, telemetry: dict):
        self.client.publish("v1/devices/me/telemetry", json.dumps(telemetry), qos=1)
        print("Published telemetry:", telemetry)

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()