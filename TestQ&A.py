import logging

from bit_agent_interface_history import Conversational
# Configure the logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
conversational = Conversational()


def run_graph(query):
    return conversational.run_graph(query)


questions = [
    # "מהו ביט?",
    # "איך מורידים את אפליקציית ביט?",
    # "האם יש עמלות על שימוש באפליקציית ביט?",
    # "איך מבצעים תשלום באמצעות ביט?",
    # "האם ניתן לקבל החזר כספי דרך אפליקציית ביט?",
    # "איך מוסיפים כרטיס אשראי לאפליקציית ביט?",
    # "האם אפליקציית ביט מאובטחת?",
    # "איך אפשר לבצע העברות כספיות בין חשבונות באמצעות ביט?",
    # "האם יש הגבלה על סכום התשלום באמצעות ביט?",
    # "איך משנים את הפרטים האישיים באפליקציית ביט?",
    # "מה היא אפליקציית ביט?",
    # "איך אני נרשם לאפליקציית ביט?",
    # "האם יש עמלות על שימוש באפליקציית ביט?",
    # "איך אני מוסיף כרטיס אשראי לאפליקציית ביט?",
    # "האם אני יכול לשלם עם ביט בחנויות?",
    # "איך אני משלם לחבר דרך אפליקציית ביט?",
    # "האם אפשר לבצע העברות כספיות בינלאומיות באמצעות ביט?",
    # "איך אני משיג את הקוד להפעלת האפליקציה?",
    # "האם יש הגבלה על סכום התשלום באמצעות ביט?",
    # "איך אני בודק את היתרה שלי באפליקציית ביט?",
    # "האם אפשר לשלם חשבונות דרך אפליקציית ביט?",
    # "איך אני מבצע חידוש סיסמה באפליקציית ביט?",
    # "מה לעשות אם נתקעתי באמצע תהליך תשלום?",
    # "איך אני מוסיף בנק לאפליקציית ביט?",
    # "האם אפשר לשלם באמצעות ביט באינטרנט?",
    # "איך אני יכול לבטל תשלום שבוצע בטעות?",
    # "האם יש תמיכה טכנית למשתמשי ביט?",
    # "איך אני מעדכן פרטי כרטיס אשראי באפליקציית ביט?",
    # "מה לעשות אם חוויתי בעיה בעת שימוש באפליקציה?",
    # "איך אני משתמש באפליקציית ביט לשליחת כסף לחשבון בנק?",
    # "האם אפליקציית ביט מאובטחת?",
    # "איך אני מגביל את ההרשאות של אפליקציית ביט?",
    # "האם אפשר לקבל החזר כספי עבור תשלום שבוצע באמצעות ביט?",
    # "איך אני יכול לראות את היסטוריית התשלומים שלי בביט?",
    # "מה היא אפליקציית ביט?",
    # "איך אני נרשם לאפליקציית ביט?",
    # "האם יש עמלות על שימוש באפליקציית ביט?",
    # "איך אני מוסיף כרטיס אשראי לאפליקציית ביט?",
    # "האם אני יכול לשלם עם ביט בחנויות?",
    # "איך אני משלם לחבר דרך אפליקציית ביט?",
    # "האם אפשר לבצע העברות כספיות בינלאומיות באמצעות ביט?",
    # "איך אני משיג את הקוד להפעלת האפליקציה?",
    # "האם יש הגבלה על סכום התשלום באמצעות ביט?",
    # "איך אני בודק את היתרה שלי באפליקציית ביט?",
    # "האם אפשר לשלם חשבונות דרך אפליקציית ביט?",
     # "what is bit app and what is the cost of transfer?",
    # "איך אני מבצע חידוש סיסמה באפליקציית ביט?",
    # "מה לעשות אם נתקעתי באמצע תהליך תשלום?",
    # "איך אני מוסיף בנק לאפליקציית ביט?",
    #  "האם אפשר לשלם באמצעות ביט באינטרנט?",
    # "איך אני יכול לבטל תשלום שבוצע בטעות?",
    # "האם יש תמיכה טכנית למשתמשי ביט?",
    # "איך אני מעדכן פרטי כרטיס אשראי באפליקציית ביט?",
    # "מה לעשות אם חוויתי בעיה בעת שימוש באפליקציה?",
    # "איך אני משתמש באפליקציית ביט לשליחת כסף לחשבון בנק?",
    # "האם אפליקציית ביט מאובטחת?",
    # "איך אני מגביל את ההרשאות של אפליקציית ביט?",
    # "האם אפשר לקבל החזר כספי עבור תשלום שבוצע באמצעות ביט?",
    # "איך אני יכול לראות את היסטוריית התשלומים שלי בביט?",
    # "מהו החומר האפל?",
    # "איך פועל האינטרנט?",
    # "מהם היתרונות והחסרונות של בינה מלאכותית?",
    # "איך ניתן להגן על הפרטיות בעידן הדיגיטלי?",
    # "מהם ההישגים המדעיים החשובים ביותר של המאה ה-21?",
    # "מהם הגורמים למהפכה הצרפתית?",
    # "מהו הרנסנס?",
    # "מהן התרבויות העתיקות המרתקות ביותר?",
    # "מהן היצירות הספרותיות החשובות ביותר בכל הזמנים?",
    # "מהם הזרמים המוזיקליים המשפיעים ביותר?",
    # "מהם הגורמים לעוני?",
    # "איך ניתן לקדם שוויון מגדרי?",
    # "מהם האתגרים העומדים בפני הדמוקרטיה בעידן המודרני?",
    # "מהן ההשלכות של גלובליזציה?",
    # "איך ניתן לפתור את משבר האקלים?",
    # "מהו משמעות החיים?",
    # "האם קיימת מציאות אובייקטיבית?",
    # "מהם ההבדלים בין טוב לרע?",
    "מהו תפקיד האמנות בחברה?",
    # "מהם היתרונות והחסרונות של חופש הביטוי?",
    # "מהי הדרך הטובה ביותר ללמוד שפה חדשה?",
    # "מהם ספרי הילדים האהובים עליכם?",
    # "מהו המתכון האהוב עליכם?",
    # "מהו המקום האהוב עליכם בעולם?",
    # "מהו החלום הגדול שלכם?",
    # "מהם הבנקים הגדולים בישראל?",
    # "מהם השירותים השונים שמציעים הבנקים בישראל?",
    # "מהם ההבדלים בין הבנקים השונים בישראל?",
    #  "איך ניתן לפתוח חשבון בנק בישראל?",
    # "מהם העמלות השונים שגובים הבנקים בישראל?",
    # "איך ניתן להתלונן על שירות לקוחות של בנק בישראל?",
    # "מהם היתרונות והחסרונות של שימוש בבנק בישראל?",
    # "מהן החלופות לשימוש בבנק בישראל?",
    # "מהם האתגרים העומדים בפני הבנקים בישראל?",
    # "מהו עתיד הבנקאות בישראל?",
    #    "מהי אפליקציית PAYBOX?",
    # "מהם השירותים השונים שמציעה אפליקציית PAYBOX?",
    # "איך ניתן להירשם לאפליקציית PAYBOX?",
    # "איך ניתן להשתמש באפליקציית PAYBOX?",
    # "מהם העמלות השונים שגובה אפליקציית PAYBOX?",
    # "איך ניתן להתלונן על שירות לקוחות של אפליקציית PAYBOX?",
    # "מהם היתרונות והחסרונות של שימוש באפליקציית PAYBOX?",
    # "מהן החלופות לשימוש באפליקציית PAYBOX?",
    # "מהם האתגרים העומדים בפני אפליקציית PAYBOX?",
    # "מהו עתיד PAYBOX?",
    # "איך ניתן להשוות בין בנקים בישראל?",
    # "איך ניתן לבחור את הבנק המתאים ביותר עבורי?",
    # "איך ניתן להשתמש באפליקציית PAYBOX בצורה בטוחה?",
    # "מהם הסיכונים הכרוכים בשימוש באפליקציית PAYBOX?",
    # "איך ניתן להגן על הפרטיות שלי בעת שימוש באפליקציית PAYBOX?"
    #
    # "מהם ההבדלים העיקריים בין בנק פועלים ובנק לאומי?",
    # "איזה בנק מציע שירותים טובים יותר ללקוחות פרטיים?",
    # "איזה בנק מציע שירותים טובים יותר ללקוחות עסקיים?",
    # "איזה בנק גובה עמלות נמוכות יותר?",
    # "איזה בנק ידוע בשירות לקוחות טוב יותר?",
    #
    # "מהם השירותים הייחודיים שמציע בנק פועלים?",
    # "מהם היתרונות והחסרונות של שימוש בבנק פועלים?",
    # "מהם האתגרים העומדים בפני בנק פועלים?",
    # "מהן תוכניות העתיד של בנק פועלים?",
    # "מהי דעתכם על בנק פועלים?",
    #
    # "מהם השירותים הייחודיים שמציע בנק לאומי?",
    # "מהם היתרונות והחסרונות של שימוש בבנק לאומי?",
    # "מהם האתגרים העומדים בפני בנק לאומי?",
    # "מהן תוכניות העתיד של בנק לאומי?",
    # "מהי דעתכם על בנק לאומי?",
    # "איזה בנק מתאים יותר לצרכים שלי?",
    # "איך ניתן לפתוח חשבון באחד מהבנקים?",
    # "איך ניתן ליצור קשר עם שירות הלקוחות של אחד מהבנקים?",
    # "מהם הסיכונים הכרוכים בשימוש באחד מהבנקים?",
    # "איך ניתן להגן על הפרטיות שלי בעת שימוש באחד מהבנקים?"
    # "כתוב תוכנית שתאפשר למשתמש להעביר כסף באמצעות ביט.",
    # "כתוב תוכנית שתאפשר למשתמש לבדוק את יתרת החשבון שלו בביט.",
    # "כתוב תוכנית שתאפשר למשתמש להציג את היסטוריית העסקאות שלו בביט.",
    # "כתוב תוכנית שתאפשר למשתמש לבקש תשלום מחבר באמצעות ביט.",
    # "כתוב תוכנית שתאפשר למשתמש לשלם עבור קנייה בחנות באמצעות ביט.",
    # "איך ניתן להשתמש ב-API של ביט כדי לפתח אפליקציות חדשות?",
    # "איך ניתן לאבטח את התשלומים באמצעות ביט?",
    # "מהם האתגרים העומדים בפני ביט?",
    # "מהו עתיד ביט?"
]

# הדפסת הרשימה
for question in questions:
    print(question)

import pandas as pd


def process_question(question):
    # This is where you'd implement your actual processing logic.
    # For demonstration, we'll just return a mock answer.
    print(question)
    try:
     answer = run_graph(question)
     print('end  -- >' + question)
    except Exception as e:
      print(f"Error processing question1 '{question}': {e}")
      answer = "error1"+str(e)
    return answer


# Process each question and collect answers
answers = [process_question(q) for q in questions]

# Create a DataFrame from the questions and answers
df = pd.DataFrame({
    'Question': questions,
    'Answer': answers
})

# Define the Excel file path (change this to your desired path)
excel_file_path = 'questions_and_answers.xlsx'

# Write the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False, engine='openpyxl')
print(df)
print(f"Excel file '{excel_file_path}' has been created with the processed questions and answers.")



