from socket import *

HOST = '127.0.0.1' # or 'localhost'
PORT = 8889
BUFSIZ =1024
ADDR = (HOST,PORT)
 
tcpCliSock = socket(AF_INET,SOCK_STREAM)
tcpCliSock.connect(ADDR)
#HWND , key up down press , keycode ,mouse click up down ,(left  right),relate x ,relate y ,abs x ,abs y ,checksum

def send_damo(HWND , key_action , keycode ,mouse_action ,mouse_button,move_time,relate_x ,relate_y ,abs_x ,abs_y ,checksum):
    global tcpCliSock
    data1 = str(HWND) + "," + key_action + "," + str(keycode) + "," +mouse_action + "," \
    + mouse_button+ ","+ str(move_time)+ "," +str(relate_x) + "," +str(relate_y) + "," +str(abs_x) + "," \
    +str(abs_y) +  "," +str(checksum)+",\n"

    #data = str(data)
    try:
        tcpCliSock.send(data1.encode())
    except Exception as e:
        try:
            tcpCliSock = socket(AF_INET,SOCK_STREAM)
            tcpCliSock.connect(ADDR)
            tcpCliSock.send(data1.encode())
        except Exception as e:
            pass
    #data1 = tcpCliSock.recv(BUFSIZ)

    #print(data1.decode('utf-8'))
    #tcpCliSock.close()

#send_damo("0","down",0,"down","right",0.2,200,210,-1,-1,"8881714191894060058")
