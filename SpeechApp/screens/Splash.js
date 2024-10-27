import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

export default function Splash({ navigation }) {
  return (
    <View style={styles.container}>
      <Image source={require('../assets/123.png')} style={styles.backgroundImage} />
      <View style={styles.topContainer}>
        <Image source={require('../assets/img.png')} style={styles.image} />
        <Text style={styles.title1}>Kural.ai</Text>
      </View>
      <View style={styles.bottomContainer}>
        <Text style={styles.title}>Get Professional Help</Text>
        <Text style={styles.subtitle}>in Professional time</Text>
        <TouchableOpacity style={styles.loginButton} onPress={() => navigation.navigate('Speech')}>
          <Text style={styles.loginText}>Let's Go</Text>
        </TouchableOpacity>
        {/* <TouchableOpacity style={styles.signUpButton} onPress={() => navigation.navigate('SignUp')}>
          <Text style={styles.signUpText}>Sign Up</Text>
        </TouchableOpacity> */}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  backgroundImage: {
    position: 'absolute',
    width: '100%',
    height: '80%',
    resizeMode: 'cover',
    opacity: 0.2,
  },
  container: {
    flex: 1,
    backgroundColor: '#50c878',
  },
  topContainer: {
    flex: 2,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomContainer: {
    width: "100%",
    flex: 1,
    paddingTop: 80,
    backgroundColor: '#fff',
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    alignItems: 'center',
    justifyContent: 'flex-start',
    padding: 20,
  },
  image: {
    width: width * 0.5,
    height: height * 0.25,
    resizeMode: 'contain',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000',
  },
  title1: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#fff',
  },
  subtitle: {
    fontSize: 16,
    color: '#000',
    marginBottom: 20,
  },
  signUpButton: {
    backgroundColor: '#50c878',
    paddingVertical: 15,
    alignItems: 'center',
    paddingHorizontal: 50,
    borderRadius: 10,
    marginBottom: 10,
    marginTop: 20,
    width: "100%",
  },
  signUpText: {
    color: '#fff',
    fontSize: 18,
  },
  loginButton: {
    backgroundColor: '#50c878',
    paddingVertical: 1,
    paddingHorizontal: 55,
    borderRadius: 10,
    paddingVertical: 15,
    alignItems: 'center',
    width: "100%",
  },
  loginText: {
    backgroundColor: '#50c878',
    color: '#fff',
    fontSize: 18,
  },
});
