with open('input.txt', 'r') as f:
    lines = f.readlines()
    first = 0
    cnt = 0
    A = []
    B = []
    for line in lines:
        if first == 0:
            A.append(int(line))
            first += 1
        elif first < 3:
            A.append(int(line))
            B.append(int(line))
            first += 1
        else:
            B.append(int(line))
            totalA = 0
            totalB = 0
            print(A, B)
            for i in range(len(B)):
                totalA += A[i]
                totalB += B[i]
            A.pop(0)
            A.append(int(line))
            B.pop(0)
            if totalA < totalB:
                cnt += 1

    print(cnt)
