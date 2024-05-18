from accelerated_trajectory import in_polygon

polygon = list(map(lambda x: list(map(float, x)), list(map(lambda x: x.split(' '), input('enter polygon: ').split('  ')))))

dec = {
    True: 'YES',
    False: 'NO'
}

while True:
    s = input('enter polygon: ')
    if s != 'same':
        polygon = list(map(lambda x: list(map(float, x)), list(map(lambda x: x.split(' '), s.split('  ')))))
    x, y = map(float, input('enter coordinates: ').split(' '))
    print(dec[in_polygon([x, y], polygon)])