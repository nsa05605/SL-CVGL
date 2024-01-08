import cv2
import numpy as np
import os
import sys

import params as params

# 비디오를 입력 받고, 이를 png 확장자의 파일로 저장함
# vid는 비디오 경로, dst는 저장될 비디오 경로
def extract_frames(vid, dst):
    print(vid)
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    print(image.shape)
    print(success)
    count = 0
    while success:
        image = cv2.resize(image, (144, 256)) # original size 720x1280
        img_path = os.path.join(dst, '%04d.png' % count)
        print(img_path)
        cv2.imwrite(img_path, image)    # save frame as JPEG file      
        success,image = vidcap.read()   # 반복
        #print('Read a new frame: ', success)
        count += 1

    return count

def master(v_list, mode):

    #pack_path =  params.raw_videos_train  
    #pack_path =  params.raw_videos_val
    
    # 여기는 들어갈 비디오 클립들 목록들 들어가는 것 같고
    videos = open(v_list, 'r')

    # 결과 저장할 파일 생성
    fname = mode + '_4.list'
    fp = open(fname, 'w')
    
    # 경로 설정
    pack_path =  params.raw_videos_train
    base_path = '/home/c3-0/shruti/data/BDD_frames/frames144_256/train/'
        
    if mode == 'val':
        pack_path =  params.raw_videos_val
        base_path = '/home/c3-0/shruti/data/BDD_frames/frames144_256/val/'

    cnt = 0
    for sample in videos:
        sample = sample.rstrip('\n')
        
        # 비디오 이름과 프레임 수 추출
        v_name, num_frames = sample.split()
        
        # 프레임 수가 1200 미만인 경우에만 처리
        if int(num_frames) < 1200:
            print(num_frames)
            print(v_name)
            cnt += 1

            # 비디오 경로 설정
            sample_path = os.path.join(pack_path, v_name+'.mov')

            # 결과 비디오 저장 경로
            dst = os.path.join(base_path, v_name+'.mov')
    
            # 비디오의 프레임 추출 및 저장하고, 결과 파일에 비디오 정보를 저장
            num_frames = extract_frames(sample_path, dst)
            fp.write('{} {}\n'.format(v_name, num_frames))
        else:
            fp.write('{}\n'.format(sample))
            pass
    
    print(cnt)

    fp.close()
    videos.close()

if __name__ == '__main__':
    v_list = None
    if len(sys.argv) > 1:
        mode = str(sys.argv[1])
    else:
        print('mode missing!')
        exit(0)

    v_list = mode + '_3.list'

    master(v_list, mode)


# 아직 모르는 부분들은,
# argv[1]에 들어가는 내용이 무엇인지 파악하기
# _3.list, _4.list 등으로 주어지는 파일이 정확히 뭔지 파악하기