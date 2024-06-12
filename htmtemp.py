# new = '''
# <style>
# h1#ChatwithPDF{
#     font-family: 'Montserrat';
#     font-size:72px;
#     color: #0a5c52;
#   }
# }
# </style>
# <style>
# h3#hello-this-is-your-pdf-chatbot{
#     padding: 0px;
# }</style>
# <style>
# [data-testid="st   FileUploadDropzone"]{
# background: rgb(10,92,82);
# background: linear-gradient(90deg, rgba(10,92,82,1) 0%, rgba(2,160,149,1) 100%);
# color: #fff;
# }
# </style>
# <style>
# [data-testid="main-menu-list"]{
#     color: fc4c4c;
# }
# </style>
# <style>
# p{
#     font-size:16px;
#     font-color: #fff;
# }
# </style>
# <style>
# .stApp {
#   background-image: url("data:image/png;base64,%s");
#   background-size: cover;
# }
# </style>
# <style>
# div.css-k3w14i effi0qh3{
#     font-size:24px;
# }
# </style>
# <style>
# button.css-aqt9oe.edgvbvh10 {
    
#     color: #02a095; 
# }
# </style>

# '''


# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         🌵
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         🎓
#     </div>    
#     <div style="padding:">
#     <div class="message">{{MSG}}</div>
#     </div>
# </div>
# '''
# background_image = """
# <style>
# body {
#     [data-testid="stAppViewContainer"]
#     background-image: url('1.jpg');
#     background-size: cover;
    
#     }
# </style>
# """
# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;
# }
# </style>
# '''



new = '''
<style>
h1#ChatwithPDF{
    font-family: 'Montserrat';
    font-size:72px;
    color: #00008B; /* White */
}
</style>
<style>
h3#hello-this-is-your-pdf-chatbot{
    padding: 0px;
    color: #ffffff; /* White */
}
</style>
<style>
[data-testid="stFileUploadDropzone"]{
    background: rgb(33, 150, 243); /* Dark blue */
    background: linear-gradient(90deg, rgba(33, 150, 243, 1) 0%, rgba(33, 150, 243, 1) 100%);
    color: #ffffff; /* White */
}
</style>
<style>
[data-testid="main-menu-list"]{
    color: #ffffff; /* White */
}
</style>
<style>
p{
    font-size:16px;
    color: #ffffff; /* White */
}
</style>
<style>
.stApp {
  background-color: #add8e6; /* Light blue */
}
</style>
<style>
div.css-k3w14i.effi0qh3{
    font-size:24px;
    color: #333333; /* Dark grey for text fields */
}
</style>
<style>
button.css-aqt9oe.edgvbvh10 {
    color: #ffffff; /* White */
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        🌵
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        🎓
    </div>    
    <div style="padding:">
    <div class="message">{{MSG}}</div>
    </div>
</div>
'''

background_image = """
<style>
body {
    [data-testid="stAppViewContainer"]
    background-color: #add8e6; /* Light blue */
    background-size: cover;
}
</style>
"""

page_bg_img = '''
<style>
body {
background-color: #add8e6; /* Light blue */
}
</style>
'''
