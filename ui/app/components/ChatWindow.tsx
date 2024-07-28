"use client";

import React, { useRef, useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { EmptyState } from "./EmptyState";
import { ChatMessageBubble, Message } from "./ChatMessageBubble";
import { AutoResizeTextarea } from "./AutoResizeTextarea";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { Flex, Box } from "@chakra-ui/react";
import { useColorMode } from "@chakra-ui/react";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { applyPatch } from "fast-json-patch";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  IconButton,
  InputGroup,
  InputRightElement,
  InputLeftElement,
  Spinner,
  CircularProgress,
  Avatar,
  VStack,
  HStack,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Spacer,
  Stack,
  Text,
} from "@chakra-ui/react";
import { ArrowUpIcon, ChevronDownIcon } from "@chakra-ui/icons";
import { MdMic, MdStop } from "react-icons/md";

import Footer from "./Footer";
import { Header } from "./Header";
import { Source } from "./SourceBubble";
import { apiBaseUrl } from "../utils/constants";
import { RiRobot2Line } from "react-icons/ri";

export function ChatWindow(props: {
  titleText: string;
  lightMode: string;
  darkMode: string;
}) {
  const { titleText, lightMode, darkMode } = props;
  const avatarStyle: React.CSSProperties = {
    width: "250px",
    height: "250px",
    objectFit: "cover",
    borderRadius: "50%",
    border: "3px solid gray",
  };
  const [isVoiceChatActive, setIsVoiceChatActive] = useState<boolean>(false);
  const { colorMode, toggleColorMode } = useColorMode();
  const toggleVoiceChat = () => setIsVoiceChatActive(!isVoiceChatActive);
  const conversationId = uuidv4();
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeechLoading, setIsSpeechLoading] = useState(false);
  const [isSpeechPlaying, setIsSpeechPlaying] = useState(false);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [chatHistory, setChatHistory] = useState<
    { human: string; ai: string }[]
  >([]);
  const [chatbots, setChatbots] = useState<string[]>([]);
  const [selectedChatbot, setSelectedChatbot] = useState<string>("");
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);
  const [userVideoStream, setUserVideoStream] = useState<MediaStream | null>(
    null,
  );
  useEffect(() => {
    if (userVideoRef.current) {
      userVideoRef.current.srcObject = userVideoStream;
    }
  }, [userVideoStream]);

  const fetchBots = async () => {
    const response = await fetch("api/bots").then((res) => res.json());
    const botNames = response.bots;
    console.log("Available chatbots:", botNames);
    setChatbots(botNames);
    setSelectedChatbot(botNames[0]);
  };
  // Load available chatbots on component mount
  useEffect(() => {
    fetchBots();
  }, []);

  // Create a ref for selected chatbot state
  const selectedChatbotRef = useRef(selectedChatbot);
  useEffect(() => {
    selectedChatbotRef.current = selectedChatbot;
  }, [selectedChatbot]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const userVideoRef = useRef<HTMLVideoElement | null>(null);

  const recorderRef = useRef({
    mediaRecorder: null as MediaRecorder | null,
    audioChunks: [] as Blob[],

    start: async function (camera: boolean = false) {
      if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        return Promise.reject(
          new Error(
            "mediaDevices API or getUserMedia method is not supported in this browser.",
          ),
        );
      }
      console.log("start recording");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: camera,
      });
      this.mediaRecorder = new MediaRecorder(stream);

      setUserVideoStream(stream); // Set the stream state

      this.mediaRecorder.ondataavailable = (event) => {
        console.log("audioChunks.length:", this.audioChunks.length);
        this.audioChunks.push(event.data);
      };
      this.mediaRecorder.start(0);
    },

    stop: async function (callback_action: null | "audio" | "video" = null) {
      console.log("stop recording");
      if (this.mediaRecorder) {
        this.mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(this.audioChunks, { type: "audio/mp3" });
          this.mediaRecorder?.stream
            .getTracks()
            .forEach((track) => track.stop()); // Stop all tracks
          this.audioChunks = [];
          setUserVideoStream(null); // Reset videoStream after stopping
          await transcribeAndSendMessage(audioBlob, callback_action);
        };
        this.mediaRecorder.stop();
      }
    },
  });

  const sendMessage = async (
    message?: string,
    noStream: boolean = false,
    callback_action: null | "audio" | "video" = null,
  ): Promise<string | undefined> => {
    // console.log("API Base Url:", apiBaseUrl);
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        id: Math.random().toString(),
        content: messageValue,
        text: messageValue,
        role: "user",
      },
    ]);
    setIsLoading(true);

    const controller = new AbortController();
    setAbortController(controller);

    let accumulatedMessage = "";
    let runId: string | undefined = undefined;
    let sources: Source[] | undefined = undefined;
    let messageIndex: number | null = null;

    let renderer = new Renderer();
    renderer.paragraph = (text) => {
      return text + "\n";
    };
    renderer.list = (text) => {
      return `${text}\n\n`;
    };
    renderer.listitem = (text) => {
      return `\n• ${text}`;
    };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code,
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });
    try {
      const sourceStepName = "FindDocs";
      let streamedResponse: Record<string, any> = {};
      if (noStream) {
        await fetch(apiBaseUrl + "/chat/ainvoke_log", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: messageValue,
            chat_history: chatHistory,
          }),
          signal: controller.signal,
        })
          .then((response) => {
            if (!response.ok) {
              throw new Error("Failed to fetch response from the server.");
            }
            return response.json();
          })
          .then((data) => {
            accumulatedMessage = data.response;
            const parsedResult = marked.parse(data.response);
            setMessages((prevMessages) => {
              let newMessages = [...prevMessages];
              newMessages.push({
                id: Math.random().toString(),
                content: parsedResult.trim(),
                text: data.response.trim(),
                runId: runId,
                sources: data.docs.map((doc: string) => ({
                  url: JSON.parse(doc).source,
                  title: JSON.parse(doc).title,
                })),
                role: "assistant",
              });
              return newMessages;
            });
          });
        setChatHistory((prevChatHistory) => [
          ...prevChatHistory,
          { human: messageValue, ai: accumulatedMessage },
        ]);
        setIsLoading(false);
      } else {
        await fetchEventSource(apiBaseUrl + "/chat/astream_log", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({
            question: messageValue,
            chat_history: chatHistory,
          }),
          signal: controller.signal,
          openWhenHidden: true,
          onerror(err) {
            throw err;
          },
          onmessage(msg) {
            if (msg.event === "end") {
              setChatHistory((prevChatHistory) => [
                ...prevChatHistory,
                { human: messageValue, ai: accumulatedMessage },
              ]);
              setIsLoading(false);
            }
            if (msg.event === "error") {
              let errorMessage = JSON.parse(msg.data).message;
              toast.error(errorMessage);
            }
            if (msg.event === "data" && msg.data) {
              const chunk = JSON.parse(msg.data);
              streamedResponse = applyPatch(
                streamedResponse,
                chunk.ops,
              ).newDocument;
              if (
                Array.isArray(
                  streamedResponse?.logs?.[sourceStepName]?.final_output
                    ?.output,
                )
              ) {
                sources = streamedResponse.logs[
                  sourceStepName
                ].final_output.output.map((doc: string) => ({
                  url: JSON.parse(doc).source,
                  title: JSON.parse(doc).title,
                }));
              }
              if (streamedResponse.id !== undefined) {
                runId = streamedResponse.id;
              }
              if (Array.isArray(streamedResponse?.streamed_output)) {
                accumulatedMessage = streamedResponse.streamed_output.join("");
              }
              const parsedResult = marked.parse(accumulatedMessage);

              setMessages((prevMessages) => {
                let newMessages = [...prevMessages];
                if (
                  messageIndex === null ||
                  newMessages[messageIndex] === undefined
                ) {
                  messageIndex = newMessages.length;
                  newMessages.push({
                    id: Math.random().toString(),
                    content: parsedResult.trim(),
                    text: accumulatedMessage.trim(),
                    runId: runId,
                    sources: sources,
                    role: "assistant",
                  });
                } else if (newMessages[messageIndex] !== undefined) {
                  newMessages[messageIndex].content = parsedResult.trim();
                  newMessages[messageIndex].text = accumulatedMessage.trim();
                  newMessages[messageIndex].runId = runId;
                  newMessages[messageIndex].sources = sources;
                }
                return newMessages;
              });
            }
          },
        });
      }
      if (callback_action === "audio") {
        playMessageAudio(accumulatedMessage, selectedChatbotRef.current);
      } else if (callback_action === "video") {
        playMessageVideo(accumulatedMessage, selectedChatbotRef.current);
      }
      return accumulatedMessage;
    } catch (e: any) {
      if (e.name === "AbortError") {
        console.log("sendMessage fetch request was aborted");
        toast.info("Fetching LLM response was canceled.");
      } else {
        toast.error(e.toString());
        throw e;
      }
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
    }
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const playMessageAudio = async (message: string, selectedChatbot: string) => {
    if (selectedChatbot === "") {
      toast.error("Please select a chatbot first.");
      return;
    }
    console.log("play message audio");
    const formData = new FormData();
    const botAudioBlob = await fetch(`/bots/${selectedChatbot}.mp3`).then(
      (res) => res.blob(),
    );
    formData.append("bot_voice_file", botAudioBlob, `${selectedChatbot}.mp3`);
    formData.append("message", message);
    formData.append("conversationId", conversationId);
    formData.append("chatbot", selectedChatbot);

    const controller = new AbortController();
    setAbortController(controller);

    setIsSpeechLoading(true);
    try {
      const audioResponse = await fetch(apiBaseUrl + "/text_to_speech", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!audioResponse.ok) {
        const errorMessage = await audioResponse.text();
        console.error("Failed to fetch audio:", errorMessage);
        toast.error(
          "Failed to transform text to speech for AI response:" + errorMessage,
        );

        setIsSpeechLoading(false);
        return;
      }

      const audioBlob = await audioResponse.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      console.log("playing audio");
      audio.play();

      audio.onplay = () => {
        console.log("audio playing");
        setIsSpeechLoading(false);
        setIsSpeechPlaying(true);
      };

      audio.onended = () => {
        console.log("audio ended");
        setIsSpeechPlaying(false);
      };

      // Handle abort controller's signal
      controller.signal.addEventListener("abort", () => {
        if (audio) {
          audio.pause();
          audio.currentTime = 0; // Reset the audio playback position
        }
        console.log("Audio playback was paused");
        toast.info("Audio playback was canceled.");
        setIsSpeechLoading(false);
        setIsSpeechPlaying(false);
      });
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Audio playback was aborted");
        toast.info("Audio playback was canceled.");
      } else {
        console.error("Audio playback error:", error);
        toast.error("Failed to play audio.");
      }
      setIsSpeechLoading(false);
    }
  };

  const playMessageVideo = async (message: string, selectedChatbot: string) => {
    if (selectedChatbot === "") {
      toast.error("Please select a chatbot first.");
      return;
    }
    console.log("play speech video");

    let formData = new FormData();
    const botAudioBlob = await fetch(`/bots/${selectedChatbot}.mp3`).then(
      (res) => res.blob(),
    );
    const botImageBlob = await fetch(`/bots/${selectedChatbot}.jpg`).then(
      (res) => res.blob(),
    );
    formData.append("bot_voice_file", botAudioBlob, `${selectedChatbot}.mp3`);
    formData.append("bot_image_file", botImageBlob, `${selectedChatbot}.jpg`);
    formData.append("message", message);
    formData.append("conversationId", conversationId);
    formData.append("chatbot", selectedChatbot);

    const controller = new AbortController();
    setAbortController(controller);

    setIsSpeechLoading(true);
    try {
      const videoResponse = await fetch(apiBaseUrl + "/text_to_video", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!videoResponse.ok) {
        const errorMessage = await videoResponse.text();
        console.error("Failed to fetch video:", errorMessage);
        toast.error(
          "Failed to transform text to video for AI response:" + errorMessage,
        );

        setIsSpeechLoading(false);
        return;
      }
      setIsVideoPlaying(true);
      const videoBlob = await videoResponse.blob();

      const videoUrl = URL.createObjectURL(videoBlob);
      if (videoRef.current) {
        videoRef.current.src = videoUrl;
        videoRef.current.play();

        videoRef.current.onplay = () => {
          console.log("video playing");
          setIsSpeechLoading(false);
          setIsSpeechPlaying(true);
        };

        videoRef.current.onended = () => {
          console.log("video ended");
          setIsSpeechPlaying(false);
          setIsVideoPlaying(false);
        };
      }

      // Handle abort controller's signal
      controller.signal.addEventListener("abort", () => {
        if (videoRef.current) {
          videoRef.current.pause();
          videoRef.current.currentTime = 0; // Reset the video playback position
        }
        console.log("Video playback was aborted");
        toast.info("Video playback was canceled.");
        setIsSpeechLoading(false);
        setIsSpeechPlaying(false);
        setIsVideoPlaying(false);
      });
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Video playback was aborted");
        toast.info("Video playback was canceled.");
      } else {
        console.error("Video playback error:", error);
        toast.error("Failed to play video.");
      }
      setIsSpeechLoading(false);
    }
  };

  const transcribeAndSendMessage = async (
    audioBlob: Blob,
    callback_action: null | "audio" | "video" = null,
  ) => {
    console.log("transcribe and send message");

    const formData = new FormData();
    formData.append("user_audio_file", audioBlob, `${conversationId}.mp3`);
    formData.append("conversationId", conversationId);

    const controller = new AbortController();
    setAbortController(controller);

    setIsTranscribing(true);
    try {
      const response = await fetch(apiBaseUrl + "/transcribe_audio", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data);
        const userMessage = data.transcript;
        setIsTranscribing(false);
        sendMessage(userMessage).then((aiMessage) => {
          if (!aiMessage) return;
          console.log("AI Message:", aiMessage);
          if (callback_action === "audio") {
            playMessageAudio(aiMessage, selectedChatbotRef.current);
          } else if (callback_action === "video") {
            playMessageVideo(aiMessage, selectedChatbotRef.current);
          }
        });
      } else {
        setIsTranscribing(false);
        toast.error("Failed to transform user audio to text.");
      }
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Transcription was aborted");
        toast.info("Transcription was canceled.");
      } else {
        console.error("Transcription error:", error);
        toast.error("Failed to transform user audio to text.");
      }
      setIsTranscribing(false);
    }
  };

  const cancelOperation = () => {
    if (abortController) {
      abortController.abort();
      setIsLoading(false);
      setIsTranscribing(false);
      setIsSpeechLoading(false);
      setIsSpeechPlaying(false);
      setIsVideoPlaying(false);

      setAbortController(null); // Clear the abort controller after aborting
    }
  };

  const toggleRecording = (
    camera: boolean = false,
    callback_action: null | "audio" | "video" = null,
  ) => {
    if (isRecording) {
      setIsRecording(false);
      recorderRef.current.stop(callback_action);
    } else {
      setIsRecording(true);
      recorderRef.current.start(camera);
    }
  };

  const noStream = false;
  const callbackAfterRecordingAudio = null;
  const callbackAfterRecordingVideo = "video";
  const callbackAfterInputtingText = null;

  return (
    <Flex
      as="main" // This makes the semantic element 'main'
      direction="column" // Stacks children vertically by default
      align="stretch" // Stretches children to fill the width
      width="full" // Ensures the Flex takes full width of its container
      minHeight="100vh" // Optional: full height of the viewport
      overflow="auto" // If you need scrolling
    >
      <Header
        titleText={titleText}
        lightMode={lightMode}
        darkMode={darkMode}
        toggleVoiceChat={toggleVoiceChat}
        isVoiceChatActive={isVoiceChatActive}
        toggleColorMode={toggleColorMode}
        colorMode={colorMode}
      />
      <Spacer />
      <Stack
        direction={["column", "row"]}
        alignItems={"center"}
        justifyContent={"center"}
        display={isVoiceChatActive ? "flex" : "none"}
      >
        <Spacer />
        <VStack>
          {isVideoPlaying ? (
            <video ref={videoRef} style={avatarStyle} autoPlay>
              Your browser does not support the video tag.
            </video>
          ) : (
            <Avatar
              style={avatarStyle}
              src={
                selectedChatbot !== ""
                  ? `/bots/${selectedChatbot}.jpg`
                  : "/images/bot.png"
              }
            />
          )}
          <HStack>
            <IconButton
              colorScheme="blue"
              rounded={"full"}
              aria-label="Bot status"
              icon={
                isSpeechPlaying ? (
                  <Spinner />
                ) : isLoading || isSpeechLoading ? (
                  <CircularProgress isIndeterminate size="30px" />
                ) : (
                  <RiRobot2Line />
                )
              }
              onClick={(e) => {
                e.preventDefault();
                if (selectedChatbot === "") {
                  toast.error("Please select a chatbot first.");
                  return;
                }
                if (isSpeechLoading || isSpeechPlaying) {
                  cancelOperation();
                  return;
                }
              }}
            />
            <Spacer />
            <Text maxWidth={"300px"} noOfLines={1}>
              {isLoading
                ? "Getting reponse..."
                : isSpeechLoading
                  ? "Generating speech..."
                  : ""}
            </Text>
          </HStack>
        </VStack>
        <Spacer />
        <VStack>
          {isRecording ? (
            <video
              ref={userVideoRef}
              autoPlay
              playsInline
              muted
              style={{
                ...avatarStyle,
                transform: "scaleX(-1)",
              }}
            />
          ) : (
            <Avatar style={avatarStyle} />
          )}
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="User mic status"
            icon={
              isRecording ? (
                <MdStop />
              ) : isTranscribing ? (
                <Spinner />
              ) : (
                <MdMic />
              )
            }
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              if (selectedChatbot === "") {
                toast.error("Please select a chatbot first.");
                return;
              }
              if (isTranscribing) {
                cancelOperation();
                return;
              }
              toggleRecording(true, callbackAfterRecordingVideo);
            }}
          />
        </VStack>
        <Spacer />
      </Stack>
      <Spacer display={isVoiceChatActive ? "flex" : "none"} />
      <Box
        display={isVoiceChatActive ? "flex" : "none"}
        flexDirection={["column", "row"]} // Stacks children vertically in reverse order.
        alignItems={"center"} // Aligns children along the center of the cross axis.
        justifyContent={"center"} // Aligns children along the center of the main axis.
        width="full" // Takes the full width of its container.
        mb={2} // Margin-bottom, where each unit is typically 4px in Chakra UI.
        p={8} // Padding on all sides, each unit is typically 4px, so 8 units equal 32px.
      >
        <Menu>
          <MenuButton as={Button} rightIcon={<ChevronDownIcon />}>
            {selectedChatbot || "Select Chatbot"}
          </MenuButton>
          <MenuList>
            {chatbots.map((bot) => (
              <MenuItem
                key={bot}
                onClick={() => {
                  console.log("Selected chatbot: ", bot);
                  setSelectedChatbot(bot);
                }}
              >
                {bot}
              </MenuItem>
            ))}
          </MenuList>
        </Menu>
      </Box>
      <Spacer display={isVoiceChatActive ? "flex" : "none"} />
      <Box
        display={isVoiceChatActive ? "none" : "flex"}
        flexDirection="column-reverse" // Stacks children vertically in reverse order.
        width="full" // Takes the full width of its container.
        mb={2} // Margin-bottom, where each unit is typically 4px in Chakra UI.
        overflow="auto" // Allows scrolling for overflow content.
        p={8} // Padding on all sides, each unit is typically 4px, so 8 units equal 32px.
        ref={messageContainerRef} // Ref for DOM access or manipulation.
      >
        {messages.length > 0 ? (
          [...messages]
            .reverse()
            .map((m, index) => (
              <ChatMessageBubble
                key={m.id}
                message={{ ...m }}
                aiEmoji="🦜"
                isMostRecent={index === 0}
                messageCompleted={!isLoading}
                selectedChatbot={selectedChatbot}
                conversationId={conversationId}
                lightMode={lightMode}
                darkMode={darkMode}
              ></ChatMessageBubble>
            ))
        ) : (
          <EmptyState
            onChoice={sendInitialQuestion}
            lightMode={lightMode}
            darkMode={darkMode}
          />
        )}
      </Box>
      <Spacer display={isVoiceChatActive ? "none" : "flex"} />
      <Box px={4} display={isVoiceChatActive ? "none" : "flex"}>
        <InputGroup size="md" alignItems={"center"}>
          <InputLeftElement h="full">
            <IconButton
              colorScheme="blue"
              rounded={"full"}
              aria-label="Send"
              icon={
                isRecording ? (
                  <MdStop />
                ) : isTranscribing ? (
                  <Spinner />
                ) : (
                  <MdMic />
                )
              }
              type="submit"
              onClick={(e) => {
                e.preventDefault();
                if (isTranscribing) {
                  cancelOperation();
                  return;
                }
                toggleRecording(false, callbackAfterRecordingAudio);
              }}
            />
          </InputLeftElement>
          <AutoResizeTextarea
            value={input}
            maxRows={5}
            marginRight={"56px"}
            marginLeft={"56px"}
            placeholder="Ask me about a patient..."
            textColor={colorMode === "light" ? "black" : "white"}
            borderColor={colorMode === "light" ? "gray.900" : "gray.100"}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage(input, noStream, callbackAfterInputtingText);
              } else if (e.key === "Enter" && e.shiftKey) {
                e.preventDefault();
                setInput(input + "\n");
              }
            }}
          ></AutoResizeTextarea>
          <InputRightElement h="full">
            <IconButton
              colorScheme="blue"
              rounded={"full"}
              aria-label="Send"
              icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
              type="submit"
              onClick={(e) => {
                e.preventDefault();
                if (isLoading) {
                  cancelOperation();
                  return;
                }
                sendMessage(input, noStream, callbackAfterInputtingText);
              }}
            />
          </InputRightElement>
        </InputGroup>
      </Box>
      <Footer colorMode={colorMode} />
    </Flex>
  );
}
