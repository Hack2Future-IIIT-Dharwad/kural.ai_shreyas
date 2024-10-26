const Stack = createNativeStackNavigator();
import * as React from "react";
import {  StyleSheet } from 'react-native';
import { NavigationContainer } from "@react-navigation/native";
import { useFonts } from "expo-font";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import Splash from "./screens/Splash";
import CustomHeader from './components/Header';
import Speech from "./screens/Speech";


const App = () => {
  const [hideSplashScreen, setHideSplashScreen] = React.useState(true);
  

  return (
    <>
      <NavigationContainer>
        {hideSplashScreen ? (
          <Stack.Navigator
            initialRouteName="Splash"
            screenOptions={{ headerShown: false }}
          >
            
            <Stack.Screen name="Splash" component={Splash} options={{ headerShown: false }} />
            
             <Stack.Screen name="Speech" component={Speech} options={({ navigation }) => ({
                headerShown: true,
                header: () => <CustomHeader navigation={navigation} title="Kural.ai" />
              })}/>
             

          </Stack.Navigator>
        ) : null}
      </NavigationContainer>
    </>
  );
};


const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
  },
  text: {
    fontSize: 24,
    color: 'orange',
  },
  backButton: {
    marginLeft: 15,
    color: 'orange',
    fontSize: 18,
  },
});


export default App;

