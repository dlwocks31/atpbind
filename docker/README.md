# Docker build steps
- 이 레포지토리의 `docker` 폴더로 이동합니다.
- 147.47.69.82의 `/home/jaechanlee/atpbind/lmg_512_4.pt`를 해당 폴더로 복사합니다.
- `docker build -t <tag> .` 으로 빌드
- `docker run -v <directory with input file>:<directory in docker> <tag> python /script.py <input pdb file> <output csv file>` 으로 스크립트를 실행하면, 입력 pdb file에 대응되는 prediction file이 output csv file에 실행
  - 예시: `docker run -v $(pwd):/data/ lmg:latest python /script.py /data/pdb/3EPSA.pdb /data/out.csv`
  - https://github.com/dlwocks31/atpbind/tree/main/data/pdb 의 pdb file을 테스트 시 사용할 수 있습니다.