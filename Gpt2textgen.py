!pip install tensorflow==1.15.2
!pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from datetime import datetime
import os

gpt2.download_gpt2(model_name="124M")

if not os.path.exists('checkpoint'):
  os.makedirs(os.path.join('checkpoint', 'run1'))
  gpt2.download_gpt2(model_name='124M')
  os.system('mv models/124M/* checkpoint/run1/')

#Load a Trained Model Checkpoint Put On Folder 'Checkpoint'
mymodel = 'mkitchen_all_stp_100'
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name=mymodel)
gentext = input("Find: ")
gpt2.generate(sess, 
    run_name=mymodel,
    temperature=0.9,
    length=373,
    top_k=40, 
    #top_p=0.9,
    prefix=gentext
    )
