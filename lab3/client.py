import socket
import json_rpc
import json
import random

def sender(method, param, id):
	result = json_rpc.JSONRPCRequest(method, param, id)
	result_packet = str(result.packet)
	print("Sent string: " + result_packet)
	client_sock.sendall(result_packet)
	client_sock.send("\n")
	print("Send.")
	data = client_sock.recv(1024)
	print('Received: ', data)
	return json.loads(data)["result"]

def createMatrix(M, N, max):
	matrix = [[random.randrange(0,max) for y in range(M)] for x in range(N)]
	return matrix

client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, proto=0)
client_sock.connect(('localhost', 8888))
print("Connected")

while True:
	enter = input('Enter everything to continue ')
	a = input('Matrix width = ')
	b = input('Matrix heigth = ')
	A = createMatrix(a, b, 50)
	B = createMatrix(a, b, 50)
	arr = [A, B]
	num = sender("SumMatrix", arr, 0)
	print("Summ matrix's A = ")
	print(A)
	print(" and B ")
	print(B)
	print(" is ")
	print(num)

client_sock.close()