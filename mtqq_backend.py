import json
import paho.mqtt.client as mqtt #import the client1
from flask import Flask
from flask_mqtt import Mqtt
from flask_socketio import SocketIO

stringToBeSent = 'aaaaaaaaaaaa'



# app = Flask(__name__)
# app.config['MQTT_BROKER_URL'] = 'test.mosquitto.org'
# app.config['MQTT_BROKER_PORT'] = 1883
# app.config['MQTT_USERNAME'] = ''
# app.config['MQTT_PASSWORD'] = ''
# app.config['MQTT_KEEPALIVE'] = 5
# app.config['MQTT_TLS_ENABLED'] = False
# app.config['MQTT_REFRESH_TIME'] = 1.0  # refresh time in seconds

# mqtt = Mqtt()



import paho.mqtt.client as mqtt #import the client1
def updateMail(gibberish):


    broker_address="test.mosquitto.org"
    client = mqtt.Client("P1") #create new instance
    client.connect(broker_address) #connect to broker
    client.subscribe("home/nodered")
    client.publish("home/nodered", name+' '+address+' '+state+' '+postalCode)

updateMail('kjdskjds,ss,s,s')



# @mqtt.on_connect()
# def handle_connect(client, userdata, flags, rc):
#     mqtt.subscribe('home/nodered')




# def create_app():
#     app = Flask(__name__)
#     mqtt.init_app(app)
# socketio = SocketIO(app)

# # @socketio.on('publish')
# def handle_publish(json_str): 
#     # data = json.loads(json_str)
#     # print('asdasd')
#     # mqtt.publish(data['name'], data[stringToBeSent])
#     mqtt.publish('home', 'hello world')

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False, debug=False)

# @mqtt.on_publish()
# def handle_publish(client, userdata, mid):
#     mqtt.publish("home/nodered","OFF")

# if __name__ == '__main__':
    # client = mqtt.Client()
    #client.username_pw_set(username, password)
    # client.on_connect = on_connect
    # client.on_message = on_message
    # client.connect('localhost')
    # client.loop_start()
    # app.run(host='0.0.0.0', port=5000)