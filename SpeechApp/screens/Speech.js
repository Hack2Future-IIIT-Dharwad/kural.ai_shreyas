import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import axios from 'axios';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

const { width, height } = Dimensions.get('window');

const Speech = ({ route }) => {
  const [sentence, setSentence] = useState('');
  const [resp, setResp] = useState('');
  const [micPressed, setMicPressed] = useState(true);
  const [imageIndex, setImageIndex] = useState(0);
  const [visemeArr, setVisemeMap] = useState([]);
  const [recording, setRecording] = useState(false);
  const [audioUri, setAudioUri] = useState(null);
  const recordingRef = useRef(null);

  useEffect(() => {
    setSentence(route.params);
  }, [route.params]);

  const synthesizeSpeech = async () => {
    try {
      const response = await axios.post(`http://viseme-server.vercel.app/viseme?inputText=${sentence}`);
      const result = response.data;
      setVisemeMap(result);

      const audioResponse = await axios.post(`http://viseme-server.vercel.app/tts?inputText=${sentence}`);
      const base64String = audioResponse.data.toString('base64');
      const fileUri = `${FileSystem.cacheDirectory}temp_audio.m4a`;
      await FileSystem.writeAsStringAsync(fileUri, base64String, { encoding: FileSystem.EncodingType.Base64 });
      
      const soundObject = new Audio.Sound();
      try {
        await soundObject.loadAsync({ uri: fileUri });
        await soundObject.setRateAsync(0.5, true);
        visemeArr.forEach((e) => {
          const duration = e.privAudioOffset / 3000;
          setTimeout(() => {
            setImageIndex(e.privVisemeId);
          }, duration / 2);
        });
        await soundObject.playAsync();
        soundObject.setOnPlaybackStatusUpdate((status) => {
          if (!status.didJustFinish) return;
          soundObject.unloadAsync();
        });
      } catch (error) {
        console.log('Error playing sound:', error);
      }
    } catch (error) {
      console.error('Error fetching or playing audio:', error);
    }
  };

  const startRecording = async () => {
    try {
      setMicPressed(!micPressed);
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        console.warn('Permission to access microphone was denied');
        return;
      }

      // Create a new instance for recording
      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
      await recording.startAsync();
      recordingRef.current = recording;
      setRecording(true);
      console.log('Recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = async () => {
    try {
      if (recordingRef.current) {
        await recordingRef.current.stopAndUnloadAsync();
        const uri = recordingRef.current.getURI();
        setAudioUri(uri);
        setRecording(false);
        console.log('Recording stopped');
        console.log('Audio URI:', uri);
        setMicPressed(!micPressed);
        // Reset recordingRef to allow a new instance next time
        recordingRef.current = null;

        const formData = new FormData();
        formData.append('file', {
          uri: uri,
          type: 'audio/x-m4a',
          name: 'recorded_audio.m4a',
        });

        try {
          const response = await axios.post('http://10.0.19.248:8000/process-audio/', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
          setResp(response);
          const base64String = response.data.encode.toString('base64');
          const fileUri = `${FileSystem.cacheDirectory}temp_response_audio.m4a`;
          await FileSystem.writeAsStringAsync(fileUri, base64String, { encoding: FileSystem.EncodingType.Base64 });

          const soundObject = new Audio.Sound();
          try {
            await soundObject.loadAsync({ uri: fileUri });
            await soundObject.playAsync();
            soundObject.setOnPlaybackStatusUpdate((status) => {
              if (!status.didJustFinish) return;
              soundObject.unloadAsync();
            });
          } catch (error) {
            console.log('Error playing sound:', error);
          }
        } catch (error) {
          console.error('Error fetching or playing audio:', error);
        }
      }
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.profileSection}>
        <Text style={styles.speechText}>{sentence}</Text>
      </View>

      {resp && (
        <View style={styles.chatContainer}>
          <View style={styles.chatBubble1}>
            <Text style={styles.chatText}>{resp.data.transcription}</Text>
          </View>
          <View style={styles.chatBubble2}>
            <Text style={styles.chatText}>{resp.data.chat}</Text>
          </View>
        </View>
      )}

      <TouchableOpacity style={styles.recordingBubble} onPress={synthesizeSpeech}>
        {micPressed ? <Icon name="record-circle" size={20} color="#50c878" /> : <Icon name="record" size={20} color="#50c878" />}
        <Text style={styles.speechText}>{micPressed ? "Click to Start" : "Click to Stop"}</Text>
      </TouchableOpacity>

      {micPressed ? (
        <TouchableOpacity style={styles.micButton} onPress={startRecording}>
          <Icon name={"microphone-outline"} size={40} color="white" />
        </TouchableOpacity>
      ) : (
        <TouchableOpacity style={styles.micButton} onPress={stopRecording}>
          <Icon name={"microphone"} size={40} color="white" />
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    padding: 20,
  },
  profileSection: {
    marginTop: 25,
    marginLeft: 20,
    marginRight: 20,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  recordingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#C3F7D4FF',
    borderRadius: 20,
    padding: 10,
    marginLeft: 10,
  },
  speechText: {
    marginLeft: 5,
  },
  chatBubble1: {
    backgroundColor: '#C3F7D4FF',
    padding: 10,
    borderRadius: 10,
    marginTop: 5,
    marginBottom: 5,
    alignSelf: 'center',
    width: '100%',

  },
  chatBubble2: {
    backgroundColor: '#B8FFEEFF',
    padding: 10,
    borderRadius: 10,
    marginTop: 5,
    marginBottom: 5,
    alignSelf: 'center',
    width: '100%',

  },
  chatContainer: {
    marginBottom: 80,
    padding: 15,
    borderRadius: 10,
    marginVertical: 5,
    alignItems: 'flex-start',
    width: '85%',
    shadowColor: '#2E8B57', // dark green shadow
    shadowOpacity: 0.2,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
  },
  chatText: {
    color: '#2F4F4F', // darker green-gray for readability
    fontSize: 16,
    lineHeight: 22,
  },
  micButton: {
    width: 150,
    height: 150,
    borderRadius: 100,
    marginTop: 150,
    top: -100,
    backgroundColor: '#50c878',
    justifyContent: 'center',
    shadowColor: '#50c878',
    shadowRadius: 10,
    elevation: 30,
    alignItems: 'center',
  },
});

export default Speech;
