import psycopg2
import hashlib
import base64

class SQL:
    def __init__(self):
        """
        This class is used to connect to a PostgreSQL database and perform various operations.
        """
        self.conn = psycopg2.connect(
            host="localhost",
            database = "test",
            user = "postgres",
            password = "ATOMATIC1328p",
            port = "5432"
        )

        self.cursor = self.conn.cursor()
    
    


    # Function to encode image file to Base64 string
    def encode_image_to_base64(image_path):
        with open(image_path, 'rb') as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_string
    
    def login(self, user : str, password : str, role: str) -> bool:
        """
        This function is used to authenticate a user by asking for their username and password.
        
        Parameters:
        cursor (cursor object): The cursor object used to execute SQL queries.
        role (int): The role of the user.
        
        Returns:
        bool: True if the login is successful, False otherwise.
        """
        if role == '1':
            role = "Committee"
        elif role == '2':
            role = "Student"
        login_state = False
        
        while not login_state:
            self.cursor.execute(f"SELECT \"username\" FROM public.\"{role}\" WHERE \"username\" = '{user}'")
            user_exists = self.cursor.fetchone()
            if user_exists is None:
                print("User not found")
                continue
            self.cursor.execute(f"SELECT \"password\" FROM public.\"{role}\" WHERE \"username\" = '{user}'")
            db_pass = self.cursor.fetchone()
            
            if db_pass is None:
                print("User not found")
                continue

            if db_pass[0] == password:
                print("Login successful")
                login_state = True
                return True
            else:
                print("Login failed")
                return False
        return False
                
    def register(self, name : str, surname : str, user : str, password : str,role: int) -> None: 
        if role == 1:
            role = "Committee"
            role_id = "CID"
        elif role == 2:
            role = "Student"
            role_id = "SID"
        self.cursor.execute(f"SELECT \"{role_id}\" FROM public.\"{role}\" ORDER BY \"{role_id}\" DESC LIMIT 1")
        last_id = self.cursor.fetchone() 
        new_id = last_id[0] + 1
        self.cursor.execute(f"INSERT INTO public.\"{role}\" (\"{role_id}\", \"name\", \"surname\", \"username\", \"password\") VALUES ({new_id},'{name}', '{surname}', '{user}', '{password}')")
        self.conn.commit()
        print("Registration successful")
    
    def get_question(self, question_id : int) -> list[str]:

        self.cursor.execute(f"SELECT \"Question\" FROM public.\"Question\" WHERE \"Q_ID\"::text SIMILAR TO '{question_id}%'")
        questions = self.cursor.fetchall()
        return questions
    
    def query_answer(self, q_id : int ,SID : int, answer : int) -> None:
        b_id = f"{str(SID)}.{str(q_id)}"
        self.cursor.execute(f"SELECT \"AID\" FROM public.\"Answering\" ORDER BY \"AID\" DESC LIMIT 1")
        last_id = self.cursor.fetchone()
        if last_id is None:
            new_id = 1
        else:
            new_id = last_id[0] + 1
        self.cursor.execute(f"INSERT INTO public.\"Answering\" (\"AID\", \"Q_ID\", \"student_answer\", \"SID\", \"BrainW_ID\") VALUES ({new_id},{q_id}, {answer}, {SID}, {b_id})")
        self.conn.commit()
        print("Answer submitted")
        
    def get_answer(self, q_id : int, SID : int) -> bool:
        self.cursor.execute(f"SELECT \"student_answer\" FROM public.\"Answering\" WHERE \"Q_ID\" = {q_id} AND \"SID\" = {SID}")
        answers = self.cursor.fetchall()
        print(answers)
        if answers is not None:
            return True
        else:
            return False
    
    
    
        


            
    # if __name__ == "__main__":
    #     # login(cursor, 2)
    #     register(cursor, 2)
