from settings import set
# Settings = set()




from preprocessing import Preprocess
df = Preprocess('file.xlsx')
df.to_excel('file_Preprocessed.xlsx')
