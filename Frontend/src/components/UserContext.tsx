import React, { createContext, useState } from 'react';

interface User {
  name: string;
  surname: string;
  email: string;
  password: string;
  role: string;
}

interface UserProviderProps {
  children: React.ReactNode;
}

const UserContext = createContext<[User | null, React.Dispatch<React.SetStateAction<User | null>> | undefined]>([null, undefined]);

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  return (
    <UserContext.Provider value={[user, setUser]}>
      {children}
    </UserContext.Provider>
  ); 
};

export default UserContext;