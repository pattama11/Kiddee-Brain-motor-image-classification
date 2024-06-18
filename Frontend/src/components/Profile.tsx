import { useContext } from 'react';
import UserContext from './UserContext';

function Profile() {
  const [user, setUser] = useContext(UserContext);

  return (
    <div>
      <h2>Welcome, {user && user.name}!</h2>
      <p>Here is your profile...</p>
    </div>
  );
}
export default Profile;