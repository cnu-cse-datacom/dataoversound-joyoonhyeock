package com.example.sound.devicesound;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

import java.util.ArrayList;
import java.util.List;

import javax.xml.transform.Result;

public class Listentone {

    int HANDSHAKE_START_HZ = 4096;
    int HANDSHAKE_END_HZ = 5120 + 1024;

    int START_HZ = 1024;
    int STEP_HZ = 256;
    int BITS = 4;

    int FEC_BYTES = 4;

    private int mAudioSource = MediaRecorder.AudioSource.MIC;
    private int mSampleRate = 44100;
    private int mChannelCount = AudioFormat.CHANNEL_IN_MONO;
    private int mAudioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private float interval = 0.1f;

    private int mBufferSize = AudioRecord.getMinBufferSize(mSampleRate, mChannelCount, mAudioFormat);

    public AudioRecord mAudioRecord = null;
    int audioEncodig;
    boolean startFlag;
    FastFourierTransformer transform;


    public Listentone() {

        transform = new FastFourierTransformer(DftNormalization.STANDARD);
        startFlag = false;
        mAudioRecord = new AudioRecord(mAudioSource, mSampleRate, mChannelCount, mAudioFormat, mBufferSize);
        mAudioRecord.startRecording();

    }

    private int findPowerSize(int round) { //주파수를 찾기 위해서 input으로 들어온 수의 제일 가까운 제곱수를 찾는 함수
        int i = 1;
        while(true) {
            if(Math.pow(2,i)  >= round)
                return (int)Math.pow(2,i);
            i++;
        }
    }
    /*
    int i = 1;
    while(true) {
    if(Math.pow(2,i)  >= round) return Math.pow(2,I);
    I++;
    }
    */

    /*
    def dominant(주요 주파수를 찾는 부분)
    푸리에 변환을 하면 허수가 나온다. 추가적으로 정규화 과정이 필요하다
    정규화 과정은 ppt링크의 source부분을 참고
    정규화를 해주는 이유? 값이 너무 크거나 작으면 적절한 값을 곱해서 값을 변경해주는..
     */
    private double findFrequency(double[] toTransform) {
        int len = toTransform.length;
        double[] real = new double[len];
        double[] img = new double[len];
        double realNum;
        double imgNum;
        double[] mag = new double[len];

        Complex[] complx = transform.transform(toTransform, TransformType.FORWARD);
        Double[] freq = this.fftfreq(complx.length, 1); //정규화 해주기 위한 부분

        /*
        w = np.fft.fft(chunk)부분?
         */

        for (int i = 0; i < complx.length; i++) {
            realNum = complx[i].getReal();
            imgNum = complx[i].getImaginary();
            mag[i] = Math.sqrt((realNum * realNum) + (imgNum * imgNum));
        }
        double peak_coeff = 0;
        int index = 0;
        for (int i = 0; i < complx.length; i++) {
            if (peak_coeff < mag[i]) {
                peak_coeff = mag[i];
                index = i;
            }
        }
        Double peak_freq = freq[index];
        return Math.abs(peak_freq * mSampleRate);
    }

    /*
    정규화 하는 과정을 java로 옮겨야 되는 부분!
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val
    */
    private Double[] fftfreq(int n, int d) { //정규화하는 과정
        double val = 1.0 / (n * d);
        int[] results = new int[n];
        Double[] double_results = new Double[n];
        int N = (n - 1) / 2 + 1;
        for (int i = 0; i <= N; i++) {
            results[i] = i;
        }
        int p2 = -(n / 2);
        for (int i = N + 1; i < n; i++) {
            results[i] = p2;
            p2--;
        }
        for (int i = 0; i < n; i++) {
            double_results[i] = results[i] * val;
        }
        return double_results;
    }

    public boolean match(double freq1, double freq2) { //def_match
        return Math.abs(freq1 - freq2) < 20;
    }

    /*
    def decode_bitchunk
    4비트씩 쪼개서 1바이트로 만들어 아스키코드를 만드는 함수
     */
    public List<Integer> decode_bitchunks(int chunk_bits, List<Integer> chunks) {
        List<Integer> out_bytes = new ArrayList<>();

        int next_read_chunk = 0;
        int next_read_bit = 0;
        int _byte = 0;
        int bits_left = 8;
        while (next_read_chunk < chunks.size()) {
            int can_fill = chunk_bits - next_read_bit;
            int to_fill = Math.min(bits_left, can_fill);
            int offset = chunk_bits - next_read_bit - to_fill;
            _byte <<= to_fill;
            int shifted = chunks.get(next_read_chunk) & (((1 << to_fill) - 1) << offset);
            _byte |= shifted >> offset;
            bits_left -= to_fill;
            next_read_bit += to_fill;
            if (bits_left <= 0) {

                out_bytes.add(_byte);
                _byte = 0;
                bits_left = 8;
            }
            if (next_read_bit >= chunk_bits) {
                next_read_chunk += 1;
                next_read_bit -= chunk_bits;
            }
        }
        return out_bytes;
    }

    public List<Integer> extract_packet(List<Double> freqs) { //def extract_packet(freqs) 함수
        List<Integer> step_bit_chucks = new ArrayList<>();
        List<Integer> bit_chucks = new ArrayList<>();
        //주파수 --> 알파벳에 대응하는 비트스트림으로??
        for (int i = 0; i < freqs.size(); i++) { //
            int temp = (int) (Math.round((freqs.get(i) - START_HZ) / STEP_HZ));
            step_bit_chucks.add(temp);
        }

        for (int i = 1; i < step_bit_chucks.size(); i++) {
            if (step_bit_chucks.get(i) > 0 && step_bit_chucks.get(i) < 16) { //[c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)] 2의 4승(16)
                bit_chucks.add(step_bit_chucks.get(i));
            }
        }
        return decode_bitchunks(BITS, bit_chucks); //8비트 아스키코드로 만들어주는 함수
    }
    //recorder로부터 음성을 읽는 코드
    public void PreRequest() {
        int blocksize = findPowerSize((int) (long) Math.round(interval / 2 * mSampleRate));
        short[] buffer = new short[blocksize];
        double[] chunk = new double[blocksize];

        List<Double> packet = new ArrayList<>(); // packet = []
        List<Integer> byte_stream = new ArrayList<>();

        while (true) {
            int bufferedReadResult = mAudioRecord.read(buffer, 0, blocksize);
            if (bufferedReadResult < 0)
                continue;
            for (int i = 0; i < blocksize; i++)
                chunk[i] = buffer[i];

            double dom = findFrequency(chunk);
            Log.d("Listentone", "freq:"+dom);

            if (startFlag && match(dom, HANDSHAKE_END_HZ)) { //HandShake_end를 만나면 종료
                byte_stream = extract_packet(packet);
                Log.d("Listentone", "original code:" + byte_stream.toString());
                String result = "";
                for (int index = 0; index < byte_stream.size(); index++) {
                    result += Character.toString((char) ((int) byte_stream.get(index)));
                }

                Log.d("Listentone", "----------------------result----------------------:" + result.toString());
                packet.clear();
                startFlag = false;
            } else if (startFlag)
                packet.add(dom); //주파수를 계속 받는 부분
            else if (match(dom, HANDSHAKE_START_HZ))
                startFlag = true;

        }
    }
}
