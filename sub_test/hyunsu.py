# TCP server example
import socket
import serial

s = serial.Serial('COM3')
while 1:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", 5000))
    server_socket.listen(5)

    print("Waiting for Message")

    client_socket, address = server_socket.accept()

    data = client_socket.recv(512).decode()
    if data == 'Q' or data == 'q':
        client_socket.close()
        break
    else:
        if data == '1' or data == '4' or data == '2' or data == '8' or data == '3':
            s.write(data.encode())

            if data == '1':
                print('control: LED ON')
            elif data == '4':
                print('control: LED OFF')
            elif data == '2':
                print('control: Speaker ON')
            elif data == '8':
                print('control: Speaker OFF')
            elif data == '3':
                print('control: Operating Motor')

        else:
            print('Input error: ', data)

    server_socket.close()

print("Disconnected... END")