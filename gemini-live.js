import { BaseProvider } from './base.js';
import WebSocket from 'ws';
import { promises as fs } from 'fs';
import path from 'path';

function writeWavHeader(buf, sampleRate, numChannels, bitsPerSample) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(buf.length + 36, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * numChannels * bitsPerSample / 8, 28);
  header.writeUInt16LE(numChannels * bitsPerSample / 8, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(buf.length, 40);
  return Buffer.concat([header, buf]);
}

export class GeminiLiveProvider extends BaseProvider {
  static get id() {
    return 'gemini-live';
  }

  static get name() {
    return 'Google Gemini Live (WebSockets)';
  }

  static get configSchema() {
    return {
      title: 'Gemini Live Service Configuration',
      fields: [
        {
          id: 'apiKey',
          label: 'Gemini API Key',
          type: 'password',
          placeholder: 'AIzaSy...',
          default: '',
          required: true
        },
        {
          id: 'model',
          label: 'Live Model Name',
          type: 'options',
          options: [
            'gemini-3.1-flash-live-preview',
            'gemini-2.5-flash-native-audio-preview-09-2025',
            'gemini-2.5-flash-native-audio-preview-12-2025'
          ],
          default: 'gemini-3.1-flash-live-preview',
          required: true
        }
      ]
    };
  }

  async executeStream(params, callbacks) {
    const { messages, systemInstruction, tools, signal, workspaceId, sessionId } = params;
    const apiKey = this.config.apiKey || process.env.GEMINI_API_KEY;
    let model = this.config.model || 'gemini-3.1-flash-live-preview';

    if (!apiKey) {
      throw new Error('API key is required for Gemini Live connection.');
    }

    if (!model.startsWith('models/')) {
      model = `models/${model}`;
    }

    const wsUrl = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=${apiKey}`;

    console.log(`[Gemini Live] Connecting to: wss://generativelanguage.googleapis.com/... (Key length: ${apiKey.length})`);
    console.log(`[Gemini Live] Targeted model formatted parameter: ${model}`);

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      const audioChunks = [];
      let audioMimeType = null;
      let isWavSaved = false;

      async function saveAccumulatedAudio() {
        if (audioChunks.length === 0 || isWavSaved) return;
        isWavSaved = true;
        const rawPcm = Buffer.concat(audioChunks);
        audioChunks.length = 0;

        let sampleRate = 24000;
        if (audioMimeType) {
          const match = audioMimeType.match(/rate=(\d+)/);
          if (match) {
            sampleRate = parseInt(match[1], 10);
          }
        }

        const wavBuffer = writeWavHeader(rawPcm, sampleRate, 1, 16);

        const sessionGeneratedDir = path.join(process.cwd(), 'workspaces', workspaceId, 'sessions', sessionId, 'generated');
        await fs.mkdir(sessionGeneratedDir, { recursive: true });

        const fileName = `audio_${Date.now()}_${Math.floor(Math.random() * 1000)}.wav`;
        const filePath = path.join(sessionGeneratedDir, fileName);
        await fs.writeFile(filePath, wavBuffer);

        const relPath = `workspaces/${workspaceId}/sessions/${sessionId}/generated/${fileName}`;
        await callbacks.onMediaPart?.({
          _localFilePath: relPath,
          mimeType: 'audio/wav'
        });
      }

      async function saveGeneratedMediaImmediately(inlineData) {
        const dataBuffer = Buffer.from(inlineData.data, 'base64');
        const mimeType = inlineData.mimeType || 'application/octet-stream';
        
        let ext = 'bin';
        if (mimeType.startsWith('image/')) ext = mimeType.split('/')[1] || 'png';
        else if (mimeType.startsWith('video/')) ext = mimeType.split('/')[1] || 'mp4';
        else if (mimeType.startsWith('audio/')) ext = mimeType.split('/')[1] || 'wav';

        const sessionGeneratedDir = path.join(process.cwd(), 'workspaces', workspaceId, 'sessions', sessionId, 'generated');
        await fs.mkdir(sessionGeneratedDir, { recursive: true });

        const typePrefix = mimeType.split('/')[0] || 'media';
        const fileName = `${typePrefix}_${Date.now()}_${Math.floor(Math.random() * 1000)}.${ext}`;
        const filePath = path.join(sessionGeneratedDir, fileName);
        await fs.writeFile(filePath, dataBuffer);

        const relPath = `workspaces/${workspaceId}/sessions/${sessionId}/generated/${fileName}`;
        await callbacks.onMediaPart?.({
          _localFilePath: relPath,
          mimeType: mimeType
        });
      }

      if (signal) {
        signal.addEventListener('abort', () => {
          console.log('[Gemini Live] Abort signal received, closing socket.');
          ws.close();
          reject(new Error('Operation aborted'));
        });
      }

      // 1. Separate history from the current turn
      const historyMessages = messages.slice(0, -1);
      const currentMessage = messages[messages.length - 1];

      // 2. Format history into a compact string (no blobs in history)
      let historyString = '';
      if (historyMessages.length > 0) {
        historyString += '\n\n=========================================\n=== CONVERSATION HISTORY ===\n=========================================\n';
        for (const msg of historyMessages) {
          const role = msg.role === 'model' ? 'Assistant' : 'User';
          let msgText = '';
          for (const part of msg.parts) {
            if (part.text) {
              msgText += part.text;
            } else if (part.thought) {
              // skip internal thinking from history
            } else if (part.functionCall) {
              msgText += `\n[Called Tool: ${part.functionCall.name} with arguments: ${JSON.stringify(part.functionCall.args)}]\n`;
            } else if (part.functionResponse) {
              msgText += `\n[Tool Response for ${part.functionResponse.name}: ${JSON.stringify(part.functionResponse.response?.result || part.functionResponse.response)}]\n`;
            } else if (part.inlineData) {
              msgText += `\n[Inline ${part.inlineData.mimeType || 'media'} data provided]\n`;
            }
          }
          historyString += `${role}: ${msgText}\n`;
        }
      }

      // 3. Pre-build liveTurns using ONLY the currentMessage, supporting multimodal parts
      const liveTurns = [];
      if (currentMessage) {
        const parts = [];

        // Prepend history as text context in the first text part
        const textParts = [];
        if (historyString) {
          textParts.push(historyString + '\n\n=== CURRENT PROMPT ===\n');
        }

        for (const part of currentMessage.parts) {
          if (part.text) {
            textParts.push(part.text);
          } else if (part.functionCall) {
            textParts.push(`\n[Called Tool: ${part.functionCall.name} with arguments: ${JSON.stringify(part.functionCall.args)}]\n`);
          } else if (part.functionResponse) {
            textParts.push(`\n[Tool Response for ${part.functionResponse.name}: ${JSON.stringify(part.functionResponse.response?.result || part.functionResponse.response)}]\n`);
          } else if (part.inlineData) {
            // Multimodal: image, video, or audio — pass through as native inlineData part
            parts.push({ inlineData: { mimeType: part.inlineData.mimeType, data: part.inlineData.data } });
          }
        }

        if (textParts.length > 0) {
          parts.unshift({ text: textParts.join('') });
        }

        if (parts.length > 0) {
          const role = currentMessage.role === 'model' ? 'model' : 'user';
          liveTurns.push({ role, parts });
        }
      }

      ws.on('open', () => {
        console.log('[Gemini Live] Connection opened successfully.');

        // Setup with AUDIO modality + thinking enabled
        const setupMsg = {
          setup: {
            model,
            generationConfig: {
              responseModalities: ['AUDIO'],
              // Enable thinking with dynamic budget (-1 = model decides)
              thinkingConfig: {
                includeThoughts: true,
                thinkingBudget: -1
              }
            }
          }
        };

        if (systemInstruction) {
          setupMsg.setup.systemInstruction = {
            parts: [{ text: systemInstruction }]
          };
        }

        if (tools && tools.length > 0) {
          setupMsg.setup.tools = [{
            functionDeclarations: tools.map(t => ({
              name: t.name,
              description: t.description,
              parameters: t.inputSchema || t.parameters
            }))
          }];
        }

        console.log('[Gemini Live] Outgoing setup frame:', JSON.stringify(setupMsg));
        ws.send(JSON.stringify(setupMsg));
      });

      ws.on('message', (data) => {
        try {
          const parsed = JSON.parse(data.toString());
          console.log('[Gemini Live] Incoming frame payload:', JSON.stringify(parsed));

          // Send history turns only AFTER server confirms setup is complete
          if (parsed.setupComplete !== undefined) {
            if (liveTurns.length > 0) {
              const clientContentMsg = {
                clientContent: {
                  turns: liveTurns,
                  turnComplete: true
                }
              };
              console.log('[Gemini Live] Outgoing clientContent turns frame:', JSON.stringify(clientContentMsg));
              ws.send(JSON.stringify(clientContentMsg));
            }
            return;
          }

          // Handle thinking and content parts
          if (parsed.serverContent?.modelTurn?.parts) {
            for (const part of parsed.serverContent.modelTurn.parts) {
              if (part.thought && part.text) {
                callbacks.onThoughtChunk?.(part.text);
              } else if (part.text) {
                callbacks.onTextChunk?.(part.text);
              } else if (part.inlineData) {
                if (part.inlineData.mimeType && part.inlineData.mimeType.startsWith('audio/pcm')) {
                  audioChunks.push(Buffer.from(part.inlineData.data, 'base64'));
                  audioMimeType = part.inlineData.mimeType;
                } else {
                  saveGeneratedMediaImmediately(part.inlineData).catch(e => {
                    console.error('[Gemini Live] Error saving media immediately:', e);
                  });
                }
              }
            }
          }

          // Handle output transcriptions (AUDIO modality text output)
          if (parsed.serverContent?.outputTranscription?.text) {
            callbacks.onTextChunk?.(parsed.serverContent.outputTranscription.text);
          }

          // Handle tool call events (top-level toolCall payload)
          if (parsed.toolCall?.functionCalls) {
            console.log('[Gemini Live] Server function call request detected:', JSON.stringify(parsed.toolCall.functionCalls));
            (async () => {
              try {
                const functionResponses = [];
                for (const call of parsed.toolCall.functionCalls) {
                  const callId = call.id || `call_${Math.random().toString(36).substring(2, 9)}`;
                  callbacks.onStepStart?.({
                    id: callId,
                    name: call.name
                  });
                  const argsStr = JSON.stringify(call.args || {});
                  callbacks.onStepDelta?.(argsStr);
                  callbacks.onStepStop?.();

                  let toolResult = {};
                  if (callbacks.onToolCall) {
                    // onToolCall returns the raw result; Gemini Live sends it via toolResponse
                    toolResult = await callbacks.onToolCall(call.name, call.args, callId);
                  }

                  // Strip inlineImage blobs from toolResponse — model doesn't need the raw bytes here
                  const responseForModel = toolResult && toolResult.inlineImage
                    ? { ...toolResult, inlineImage: { mimeType: toolResult.inlineImage.mimeType, data: '[binary stripped]' } }
                    : toolResult;

                  functionResponses.push({
                    id: callId,
                    name: call.name,
                    response: { result: responseForModel }
                  });
                }

                const toolResponseMsg = {
                  toolResponse: {
                    functionResponses
                  }
                };
                console.log('[Gemini Live] Outgoing toolResponse frame:', JSON.stringify(toolResponseMsg));
                ws.send(JSON.stringify(toolResponseMsg));
              } catch (err) {
                console.error('[Gemini Live] Tool execution error:', err);
                ws.close();
                reject(err);
              }
            })();
            return;
          }

          if (parsed.serverContent?.turnComplete) {
            console.log('[Gemini Live] Server turnComplete flag received.');
            saveAccumulatedAudio().then(() => {
              ws.close();
              resolve();
            }).catch(reject);
          }
        } catch (err) {
          console.error('[Gemini Live] Error parsing incoming socket frame:', err);
          ws.close();
          reject(err);
        }
      });

      ws.on('error', (err) => {
        console.error('[Gemini Live] WebSocket client error connection event:', err);
        reject(err);
      });

      ws.on('close', (code, reason) => {
        console.log(`[Gemini Live] WebSocket closed (Code: ${code}, Reason: ${reason || 'none'}).`);
        saveAccumulatedAudio().then(() => {
          resolve();
        }).catch(reject);
      });
    });
  }

  /**
   * Send real-time audio input (raw PCM, 16kHz) via an active session.
   * Call this after executeStream opens a connection.
   */
  sendAudio(ws, pcmBuffer) {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      realtimeInput: {
        audio: {
          data: pcmBuffer.toString('base64'),
          mimeType: 'audio/pcm;rate=16000'
        }
      }
    }));
  }

  /**
   * Send a JPEG video frame via an active session.
   */
  sendVideoFrame(ws, jpegBuffer) {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      realtimeInput: {
        video: {
          data: jpegBuffer.toString('base64'),
          mimeType: 'image/jpeg'
        }
      }
    }));
  }

  /**
   * Send an image inline to be included as context.
   */
  sendImage(ws, imageBuffer, mimeType = 'image/png') {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      realtimeInput: {
        video: {
          data: imageBuffer.toString('base64'),
          mimeType
        }
      }
    }));
  }
}
