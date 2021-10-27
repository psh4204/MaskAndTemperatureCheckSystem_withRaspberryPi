import board
import busio as io
import adafruit_mlx90614

from time import sleep

i2c = io.I2C(board.SCL, board.SDA, frequency=10000) # 10k ~ 100k Hz
mlx = adafruit_mlx90614.MLX90614(i2c)

while True:
    # ToDo: 센서가 좋지않아서 손을좀 봐줘야한다.. 
    # 거리는 1cm에서 가장 좋게 나온다.
    # 적외선을 빨대같은걸로 모아보자.
    # 점퍼선 때문에 문제가 생길 수 있다. 
    # 자꾸 왔다갔다 거려 이거;;
    ambientTemp = "{:.2f}".format(mlx.ambient_temperature)
    targetTemp = "{:.2f}".format(mlx.object_temperature*0.81)

    sleep(0.1)

    print("Ambient Temperature:", ambientTemp, "°C")
    print("Target Temperature:", targetTemp,"°C")
    print("---------------------------------------")